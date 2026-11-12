# 分阶段 LoRA 训练

所有参数直接在 config.py 的 argparse 默认值中修改，也可通过命令行覆盖。
每次启动会将 Python 终端标准输出、标准错误同步写入 `output_dir/logs/<阶段>_<时间>_<进程号>.log`，不覆盖旧日志。
日志包含启动参数、环境版本、GPU 信息、数据路径及超长跳过统计、训练与验证指标、阶段起止和未捕获的 Python 异常堆栈。
启动时会打印日志绝对路径；`--log_dir` 可指定目录，`--log_level` 控制 Python logging 的级别，训练指标频率仍由 logging_steps 控制。
日志即时刷新。进程被 SIGKILL、系统 OOM killer 杀死或发生原生库崩溃时，Python 无法保证输出完整堆栈；已写出的日志仍保留。
train.py 负责模型加载和阶段开关，各阶段有独立模块：
- train_dae.py：DAE 重建训练。
- train_sft.py：结构加权 MR-SFT。
- train_dpo.py：DPO 偏好训练。

每个 train_*.py 文件都包含自己的损失函数、Trainer 类、验证循环、训练参数和保存流程。
DAE 的重建损失在 train_dae.py，MR-SFT 的加权损失在 train_sft.py，DPO 的偏好损失和 reference 缓存在 train_dpo.py。
trainer.py 已移除；data_loader.py 和 tokenize_utils.py 只负责通用数据加载、权重准备、分词和组批。
没有自动多轮调用，每一步由你单独运行验证。

## 安装

Python 3.10/3.11，单 NVIDIA GPU；先安装与服务器 CUDA 匹配的 PyTorch，再安装 requirements.txt 中的依赖。

```bash
pip install -r requirements.txt
```

在 config.py 中配置 model_path、data_dir、output_dir、gpu_id 和各阶段 *_files。
*_files 中的相对文件名基于 data_dir；支持绝对路径、多个 JSON/JSONL 文件。
数据默认文件名只是可修改的占位值，需与你服务器上的文件一致。

验证集在 config.py 中分别配置 `dae_val_files`、`sft_val_files`、`dpo_val_files`，
相对路径同样基于 data_dir，格式与对应阶段训练集一致。默认文件名为：

- codenet_dae_objc_val.jsonl
- codenet_bt_java_val_CS.jsonl
- codenet_dpo_java_val.jsonl

这些是占位文件名，请改为实际验证集路径；文件缺失会报错，不会静默跳过。
默认 `eval_strategy="epoch"`，每个 epoch 结束计算并记录 eval_loss。
若要按步验证，设置 `eval_strategy="steps"` 和 `eval_steps`。
验证使用独立数据，不进行反向传播；loss 按题数平均。
DPO 对训练集、验证集分别缓存固定 reference 的概率，验证缓存为 val_reference_logps.pt。
断点恢复需保留两份缓存，并保持训练集和验证集一致。
这里的验证指标为目标函数 loss，不代表编译或单元测试通过率。

## 单独运行各阶段

从基础模型创建新的 LoRA，只跑 DAE：

```bash
python train.py --skip_sft --skip_dpo
python train.py --skip_dae --skip_dpo
python train.py --skip_dae --skip_sft 
```

加载 DAE 权重，只跑结构加权 MR-SFT：

```bash
python train.py --skip_dae --skip_dpo --resume_from_checkpoint /你的输出目录/dae/final
```

加载 SFT 权重，只跑 DPO：

```bash
python train.py --skip_dae --skip_sft --resume_from_checkpoint /你的输出目录/sft/final
```

也可以直接从基础模型只跑 SFT：

```bash
python train.py --skip_dae --skip_dpo
```

全部阶段按顺序执行：`python train.py`。
各阶段 final 权重保存到 output_dir/dae/final、output_dir/sft/final、output_dir/dpo/final。
旧权重必须为匹配 model_path 的 PEFT adapter。完整模型/合并模型通过 model_path 指定。
resume_from_checkpoint 保持旧框架语义：只加载 LoRA 权重，重新创建优化器。
恢复训练步数、优化器等使用独立参数，且只开启一个阶段：

```bash
python train.py --skip_dae --skip_dpo --trainer_checkpoint /你的输出目录/sft/checkpoint-100
```

DPO 断点恢复必须保留 checkpoint 上一级的 reference_logps.pt 和 val_reference_logps.pt，使用相同数据和参数。
每次新训练请设置新的 output_dir，避免覆盖已有实验。

## 数据要求

DAE：instruction、input（已加噪）、output（原始代码）。不在训练阶段额外生成噪声。
SFT：instruction、input、references 数组，每个 reference 包含 output。
默认 `weight_mode=java_objc_ast`，直接读取每个 reference 中预计算的 `structure_distance`，它表示输入 Java 到该 ObjC 的归一化结构距离。
先按输出去重，再按距离从小到大取最多 `max_references` 个（默认 4）；不足 4 个则全部使用。
对选中参考计算 `softmax(-tau * structure_distance)`，距离越小权重越大，题内权重和为 1，单参考权重为 1。
训练时不重新解析 AST，不计算 ObjC 之间的距离；原数据中的 weighting 元数据用于记录，实际 tau 和参考上限由 config.py 控制。
`--weight_mode uniform` 可运行等权基线；有距离时仍选最近 N 个，无距离的旧格式则取前 N 个。
结构相似性不代表正确性：references 应由你预先筛选，确保来自同一任务且通过测试。
DPO：instruction、input、chosen、rejected、negative_type（semantic 或 syntax）。chosen 必须来自真实测试通过的样本。
默认 SFT 使用序列 log probability 求和；--length_normalize 可改成 token 平均。
按题合并相同 instruction/input、去重 reference；训练 batch size 表示题数，实际序列数量是题内参考数之和。
AST 解析通过不能代替编译和单元测试；本框架未实现生成候选或执行测试程序。

只检查某阶段数据（不加载模型）：


```bash
python train.py --skip_dae --skip_dpo --check_data
```

本机只完成数据逻辑测试与语法检查，未进行真实 CUDA 训练。各阶段需在你的服务器逐步验证。
