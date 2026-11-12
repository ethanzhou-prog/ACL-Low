# from pathlib import Path
# import shutil

# root = Path("/data/chaoni/zzy/ACL-Low/dataset/java-ast-clustered-jsonl")
# output_dir = Path("/Users/zhiyangzhou/Desktop/ACL-Low/dataset")
# output_dir.mkdir(parents=True, exist_ok=True)

# output_file = output_dir / "codenet.jsonl"

# files = sorted(root.rglob("*.jsonl"))
# total = 0

# with open(output_file, "wb") as out:
#     for file in files:
#         with open(file, "rb") as f:
#             for line in f:
#                 if line.strip():
#                     out.write(line if line.endswith(b"\n") else line + b"\n")
#                     total += 1

# print(f"合并文件数量: {len(files)}")
# print(f"样本总数量: {total}")
# print(f"输出路径: {output_file}")

from pathlib import Path

root = Path("/data/chaoni/zzy/ACL-Low/dataset")

total = 0
files = list(root.rglob("*.jsonl"))

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        total += sum(1 for line in f if line.strip())

print(f"JSONL 文件总数: {len(files)}")
print(f"样本总数量: {total}")