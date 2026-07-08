"""
Reads the design-ID CSV, validates entries, and produces a reproducible
sample for pipeline development/testing.

Kept separate from the fetch logic on purpose: "which IDs to process"
should stay independent of "how to fetch them" so the sampling strategy
can change later (e.g. stratified by category) without touching fetch code.
"""

import argparse
import csv
import random

import config


def load_design_ids(csv_path, id_column="design_id"):
    """
    Reads the CSV and returns a list of valid integer design IDs.
    Rows with missing/non-integer values are skipped and reported,
    not silently dropped -- this is our first "incomplete/inconsistent
    data" handling point in the pipeline.
    """
    valid_ids = []
    bad_rows = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Expected column '{id_column}' not found. "
                f"Columns present: {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=2):  # start=2 -> account for header row
            raw_value = row.get(id_column, "").strip()
            try:
                valid_ids.append(int(raw_value))
            except (ValueError, TypeError):
                bad_rows.append((i, raw_value))

    if bad_rows:
        print(f"[sampler] Skipped {len(bad_rows)} malformed row(s):")
        for line_num, val in bad_rows[:10]:  # only show first 10 to avoid flooding console
            print(f"    line {line_num}: '{val}'")
        if len(bad_rows) > 10:
            print(f"    ... and {len(bad_rows) - 10} more")

    # De-duplicate while preserving order, in case the CSV has repeats
    seen = set()
    deduped = []
    for did in valid_ids:
        if did not in seen:
            seen.add(did)
            deduped.append(did)

    if len(deduped) != len(valid_ids):
        print(f"[sampler] Removed {len(valid_ids) - len(deduped)} duplicate ID(s)")

    return deduped


def get_sample(all_ids, sample_size):
    """
    Returns a reproducible random sample using the fixed seed in config.py.
    If sample_size >= len(all_ids), returns all IDs (useful for full-catalog runs).
    """
    rng = random.Random(config.RANDOM_SEED)
    if sample_size >= len(all_ids):
        print(f"[sampler] Requested sample_size >= total IDs available; using all {len(all_ids)}")
        return all_ids
    return rng.sample(all_ids, sample_size)


def main():
    parser = argparse.ArgumentParser(description="Sample design IDs from the CSV for pipeline processing.")
    parser.add_argument("csv_path", help="Path to the design_ids.csv file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.DEFAULT_SAMPLE_SIZE,
        help=f"Number of IDs to sample (default: {config.DEFAULT_SAMPLE_SIZE}). Use a large number for full-catalog run.",
    )
    parser.add_argument(
        "--id-column",
        default="design_id",
        help="Name of the column containing design IDs (default: designId)",
    )
    args = parser.parse_args()

    all_ids = load_design_ids(args.csv_path, args.id_column)
    print(f"[sampler] Loaded {len(all_ids)} valid unique design IDs from CSV")

    sample = get_sample(all_ids, args.sample_size)
    print(f"[sampler] Selected sample of {len(sample)} design ID(s)")

    out_path = f"{config.DATA_DIR}/sampled_ids.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(str(i) for i in sample))
    print(f"[sampler] Saved sample to {out_path}")


if __name__ == "__main__":
    main()