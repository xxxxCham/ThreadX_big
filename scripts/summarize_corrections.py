import csv
from collections import Counter
from pathlib import Path


def main():
    p = Path("artifacts/pandera_correction_plan.csv")
    if not p.exists():
        print("Missing artifacts/pandera_correction_plan.csv")
        return

    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        total = 0
        applied = 0
        errs = Counter()
        actions = Counter()
        by_type = Counter()
        samples_failed = []
        for row in r:
            total += 1
            if row.get("applied", "").strip().lower() == "true":
                applied += 1
            et = row.get("errors", "").strip()
            if et:
                errs[et] += 1
                if len(samples_failed) < 50:
                    samples_failed.append(row)
            a = row.get("actions", "").strip()
            if a:
                for act in a.split(";"):
                    actions[act] += 1
            t = row.get("type", "").strip().lower()
            if t:
                by_type[t] += 1

    print(f"TOTAL: {total}")
    print(f"APPLIED: {applied}")
    print(f"FAILED: {total-applied}\n")

    print("Top 12 error messages (truncated):")
    for k, v in errs.most_common(12):
        print(v, ":", (k[:240].replace("\n", " ")))

    print("\nTop 12 actions:")
    for k, v in actions.most_common(12):
        print(v, k)

    print("\nCounts by type:")
    for k, v in by_type.most_common():
        print(k, v)

    # write failed rows CSV
    out = Path("artifacts/pandera_failed_files.csv")
    with out.open("w", newline="", encoding="utf-8") as of:
        writer = csv.DictWriter(
            of, fieldnames=["source", "dest", "type", "errors", "actions"]
        )
        writer.writeheader()
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("errors", "").strip():
                    writer.writerow(
                        {
                            k: row.get(k, "")
                            for k in ["source", "dest", "type", "errors", "actions"]
                        }
                    )

    if samples_failed:
        print("\nExample failed rows (up to 10):")
        for r in samples_failed[:10]:
            print("-", r.get("source"), "->", r.get("errors")[:200].replace("\n", " "))

    print(f"Wrote failed list to {out}")


if __name__ == "__main__":
    main()
