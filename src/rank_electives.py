"""
Beginner-friendly analysis script:
- Reads a graduate exit survey Excel file from the repository.
- Cleans and reshapes elective-course rating data.
- Produces a rank-ordered table of electives based on average ratings.
- Creates one bar chart figure from the ranking.
- Saves all outputs into the outputs/ folder.

This script is intentionally verbose and heavily commented for learning purposes.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# 1) Define file locations
# -----------------------------
# We build paths relative to this script so the workflow works the same on any machine.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "exit_survey_2024.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs"

RANKING_CSV = OUTPUT_DIR / "elective_course_ranking.csv"
RANKING_PNG = OUTPUT_DIR / "elective_course_ranking.png"
SUMMARY_TXT = OUTPUT_DIR / "run_summary.txt"


# -----------------------------
# 2) Helper function
# -----------------------------
def extract_course_name(column_name: str) -> str:
    """
    Convert a long survey column label into a short course name.

    Example input:
      "Rate ACC 6020 Advanced Financial Application ... - ACC 6020 Advanced Financial Application"
    Example output:
      "ACC 6020 Advanced Financial Application"

    Strategy:
    - If there is " - ACC ..." in the text, use that trailing part.
    - Otherwise, find the first "ACC #### ..." pattern.
    - If both fail, return the original column name.
    """
    dash_match = re.search(r"-\s*(ACC\s*\d+[A-Z]*\s+.+)$", column_name, flags=re.IGNORECASE)
    if dash_match:
        return " ".join(dash_match.group(1).split())

    acc_match = re.search(r"(ACC\s*\d+[A-Z]*\s+.+)$", column_name, flags=re.IGNORECASE)
    if acc_match:
        return " ".join(acc_match.group(1).split())

    return column_name.strip()


# -----------------------------
# 3) Main workflow
# -----------------------------
def main() -> None:
    # Ensure output folder exists so writing files never fails due to a missing directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Step 1/6: Loading exit survey dataset from the repository")
    print("=" * 72)
    print(f"Data file path: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found at {DATA_FILE}. "
            "Please verify the file exists in data/."
        )

    # The Excel export has:
    # - Row 1: short variable names (Q75, Q76_1, etc.)
    # - Row 2: human-readable question text
    # - Row 3: import metadata
    # - Row 4+: actual responses
    # We use header=1 so the readable question text becomes column names,
    # and skip row 3 (index 2 in zero-based counting).
    survey_df = pd.read_excel(DATA_FILE, header=1, skiprows=[2])

    print(f"Loaded dataset shape: {survey_df.shape[0]} rows x {survey_df.shape[1]} columns")

    print("\n" + "=" * 72)
    print("Step 2/6: Selecting elective rating columns")
    print("=" * 72)

    # We identify columns that contain both:
    # - The word "Rate" (these are rating questions)
    # - An elective-course pattern like "ACC ####"
    # This keeps selection deterministic and transparent.
    elective_rating_columns = [
        col
        for col in survey_df.columns
        if isinstance(col, str)
        and "rate" in col.lower()
        and re.search(r"acc\s*\d+", col, flags=re.IGNORECASE)
    ]

    if not elective_rating_columns:
        raise ValueError(
            "No elective rating columns were identified. "
            "Please inspect the survey headers."
        )

    print(f"Found {len(elective_rating_columns)} elective rating columns:")
    for col in elective_rating_columns:
        print(f"  - {extract_course_name(col)}")

    print("\n" + "=" * 72)
    print("Step 3/6: Cleaning numeric ratings (expected scale: 1 to 5)")
    print("=" * 72)

    # Work only with the elective rating columns.
    ratings_wide_df = survey_df[elective_rating_columns].copy()

    # Convert each course column to numeric.
    # Any non-numeric values become NaN, which we safely ignore later.
    for col in ratings_wide_df.columns:
        ratings_wide_df[col] = pd.to_numeric(ratings_wide_df[col], errors="coerce")

    # Optional quality safeguard: keep only values in the 1-5 range.
    # Out-of-range values are treated as missing.
    ratings_wide_df = ratings_wide_df.where((ratings_wide_df >= 1) & (ratings_wide_df <= 5))

    print("Converted rating columns to numeric and enforced valid range [1, 5].")

    print("\n" + "=" * 72)
    print("Step 4/6: Reshaping data to a long format for easy ranking")
    print("=" * 72)

    # Wide format = one row per student with many course columns.
    # Long format = one row per (student, course rating) pair.
    ratings_long_df = ratings_wide_df.melt(
        var_name="raw_course_column",
        value_name="rating"
    )

    # Remove missing ratings so calculations use only real responses.
    ratings_long_df = ratings_long_df.dropna(subset=["rating"]).copy()

    # Add clean course names.
    ratings_long_df["course"] = ratings_long_df["raw_course_column"].apply(extract_course_name)

    print(f"Long-format records with valid ratings: {len(ratings_long_df)}")

    print("\n" + "=" * 72)
    print("Step 5/6: Building rank-ordered elective results")
    print("=" * 72)

    # Aggregate by course and compute summary metrics.
    ranking_df = (
        ratings_long_df.groupby("course", as_index=False)
        .agg(
            average_rating=("rating", "mean"),
            response_count=("rating", "count")
        )
    )

    # Sort by:
    #   1) Highest average rating first
    #   2) If tied, more responses first
    #   3) If still tied, alphabetical course name
    # This creates deterministic ordering.
    ranking_df = ranking_df.sort_values(
        by=["average_rating", "response_count", "course"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # Convert to 1-based rank position.
    ranking_df.insert(0, "rank", ranking_df.index + 1)

    # Round averages for readability in final output.
    ranking_df["average_rating"] = ranking_df["average_rating"].round(3)

    # Save the ranking table.
    ranking_df.to_csv(RANKING_CSV, index=False)

    print(f"Saved ranking table to: {RANKING_CSV}")

    print("\nTop ranked electives:")
    print(ranking_df.head(10).to_string(index=False))

    print("\n" + "=" * 72)
    print("Step 6/6: Creating one figure (bar chart of average ratings)")
    print("=" * 72)

    # Plot in ranked order.
    plt.figure(figsize=(12, 6))
    plt.bar(ranking_df["course"], ranking_df["average_rating"])
    plt.title("Elective Course Ranking by Average Student Rating")
    plt.xlabel("Elective Course")
    plt.ylabel("Average Rating (1-5)")
    plt.ylim(0, 5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RANKING_PNG, dpi=150)
    plt.close()

    print(f"Saved ranking figure to: {RANKING_PNG}")

    # Write a plain-text summary for quick review and grading context.
    summary_lines = [
        "Elective Course Ranking Run Summary",
        "-" * 40,
        f"Input dataset: {DATA_FILE.name}",
        f"Total survey rows loaded: {survey_df.shape[0]}",
        f"Elective rating columns used: {len(elective_rating_columns)}",
        f"Valid rating records analyzed: {len(ratings_long_df)}",
        "",
        "Top 5 electives by average rating:",
    ]

    for _, row in ranking_df.head(5).iterrows():
        summary_lines.append(
            f"  #{int(row['rank'])}: {row['course']} "
            f"(avg={row['average_rating']}, n={int(row['response_count'])})"
        )

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved run summary to: {SUMMARY_TXT}")

    print("\nWorkflow complete. Outputs are ready in outputs/.")


if __name__ == "__main__":
    main()
