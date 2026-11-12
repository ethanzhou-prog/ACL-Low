import json
import math
from pathlib import Path

from tokenize_utils import SequenceTooLongError, encode, prompt_text


def data_config(args):
    return {
        "model": {
            "max_seq_length": args.max_seq_length,
        },
        "sft": {
            "weight_mode": args.weight_mode,
            "tau": args.tau,
            "max_references": args.max_references,
        },
        "dpo": {
            "semantic_weight": args.semantic_weight,
            "syntax_weight": args.syntax_weight,
        },
    }


def load_stage_dataset(args, tokenizer, stage, split="train"):
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")

    field = stage + ("_val_files" if split == "val" else "_files")
    paths = [
        str(Path(args.data_dir) / name)
        for name in getattr(args, field)
    ]

    print(f"[{stage}/{split}] Loading: {paths}")
    return prepare(read_rows(paths), stage, data_config(args), tokenizer)


def read_rows(paths):
    rows = []

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            if Path(path).suffix == ".json":
                data = json.load(handle)
            else:
                data = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]

        if not isinstance(data, list):
            raise ValueError(f"{path}: expected array or JSONL")

        rows.extend(data)

    return rows


def merge_tasks(rows):
    tasks = {}

    for row in rows:
        key = (row["instruction"], row.get("input", ""))
        task = tasks.setdefault(key, dict(row, references=[]))

        refs = row.get("references") or (
            [{"output": row["output"]}]
            if row.get("output")
            else []
        )

        task["references"].extend(
            dict(ref) if isinstance(ref, dict) else {"output": ref}
            for ref in refs
        )

    for task in tasks.values():
        task["references"] = list(
            {
                ref["output"]: ref
                for ref in task["references"]
            }.values()
        )

    return list(tasks.values())


def structure_distance(ref):
    distance = ref.get("structure_distance")

    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
        or not 0 <= distance <= 1
    ):
        raise ValueError(
            "Each reference needs a finite Java-to-ObjC "
            "structure_distance in [0, 1]"
        )

    return distance


def select_references(refs, cfg):
    if not refs:
        return []

    limit = cfg.get("max_references", 4)
    if limit < 1:
        raise ValueError("max_references must be positive")

    # Uniform mode supports legacy data without distance fields.
    if (
        cfg["weight_mode"] == "uniform"
        and not all("structure_distance" in ref for ref in refs)
    ):
        return refs[:limit]

    return sorted(refs, key=structure_distance)[:limit]


def reference_weights(refs, cfg):
    if not refs:
        raise ValueError("Cannot calculate weights for empty references")

    if cfg["weight_mode"] == "uniform":
        return [1 / len(refs)] * len(refs)

    if cfg["weight_mode"] != "java_objc_ast":
        raise ValueError("Unknown weight_mode")

    tau = cfg["tau"]
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")

    # Read precomputed Java-to-ObjC distances.
    logits = [
        -tau * structure_distance(ref)
        for ref in refs
    ]
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)

    return [value / total for value in values]


def prepare(rows, stage, cfg, tokenizer=None):
    items = []
    skipped = 0
    no_positive = 0

    if stage == "sft":
        rows = merge_tasks(rows)

    for task_index, row in enumerate(rows, 1):
        prompt = prompt_text(row)

        if stage == "sft":
            refs = row["references"]

            if not refs:
                no_positive += 1
                print(
                    "[sft/Data] Skipping task: references/output are empty; "
                    f"task_index={task_index}, label={row.get('label')!r}"
                )
                # print(
                #     "[sft/Data] Skipped record: "
                #     + json.dumps(row, ensure_ascii=False)
                # )
                continue

            refs = select_references(refs, cfg["sft"])
            row = dict(row, references=refs)

            outputs = [ref["output"] for ref in refs]
            weights = reference_weights(refs, cfg["sft"])

        elif stage == "dpo":
            outputs = [row["chosen"], row["rejected"]]

            if outputs[0] == outputs[1]:
                raise ValueError("chosen and rejected must differ")

            kind = row.get("negative_type", "semantic")
            if kind not in ("semantic", "syntax"):
                raise ValueError(
                    "negative_type must be semantic or syntax"
                )

            weights = [cfg["dpo"][kind + "_weight"]] * 2

        else:
            outputs = [row["output"]]
            weights = [1.0]

        if any(
            not isinstance(output, str) or not output.strip()
            for output in outputs
        ):
            raise ValueError("Empty completion")

        if tokenizer:
            try:
                sequences = [
                    encode(
                        tokenizer,
                        prompt,
                        output,
                        cfg["model"]["max_seq_length"],
                    )
                    for output in outputs
                ]
            except SequenceTooLongError:
                # Keep references/pairs intact when skipping overlong tasks.
                skipped += 1
                continue

            items.append({
                "sequences": sequences,
                "weights": weights,
            })
        else:
            items.append(row)

    if tokenizer:
        print(
            f"[{stage}/Tokenize] Kept {len(items)}/{len(rows)} tasks; "
            f"skipped {skipped} over "
            f"max_seq_length={cfg['model']['max_seq_length']}; "
            f"skipped {no_positive} without references."
        )
    elif stage == "sft":
        print(
            f"[sft/Data] Kept {len(items)}/{len(rows)} tasks; "
            f"skipped {no_positive} without references."
        )

    if not items:
        raise ValueError(
            f"No usable {stage} data: total_tasks={len(rows)}, "
            f"no_positive={no_positive}, overlength={skipped}"
        )

    return items