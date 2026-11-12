import json
from pathlib import Path


file_path = Path(
    "/data/chaoni/zzy/ACL-Low/dataset/codenet-1000-demo.jsonl"
)
new_instruction = "\nTranslate the Java code to TypeScript."

records = []

with file_path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"第 {line_number} 行不是合法 JSON"
            ) from exc

        record["instruction"] = new_instruction
        records.append(record)

# 直接覆盖原文件，不创建新的输出文件
with file_path.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"修改完成，共更新 {len(records)} 条数据：{file_path}")