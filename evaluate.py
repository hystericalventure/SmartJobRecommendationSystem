"""
evaluate.py
-----------
Simple analysis of the trained system. There is NO ground truth (no labelled
"correct" recommendations), so we do NOT compute accuracy. We report real
measurements taken from your model and dataset.

Run it with:
    python3 evaluate.py
It prints the report and also saves it to evaluation_report.txt
"""

import time

import numpy as np

from recommender import load_artifacts, recommend_jobs

TEST_PROFILES = {
    "Test 1": "Python, Machine Learning, SQL",
    "Test 2": "Java, Spring Boot, MySQL",
    "Test 3": "Data Analysis, Excel, SQL, Power BI",
    "Test 4": "NLP, Python, Machine Learning",
}


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    art = load_artifacts()
    jobs = art["jobs"]

    log("=== DATASET / MODEL STATISTICS ===")
    log(f"Jobs processed:            {art['job_vectors'].shape[0]}")
    log(f"Vocabulary size:           {len(art['vectorizer'].vocabulary_)}")
    log(f"Non-empty descriptions:    {int((jobs['description'] != '').sum())}")
    log(f"Non-zero values in matrix: {art['job_vectors'].nnz}")

    log("\n=== TEST PROFILES ===")
    all_avg = []
    for name, skills in TEST_PROFILES.items():
        start = time.perf_counter()
        results = recommend_jobs(skills, top_n=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log(f"\n{name}: {skills}")
        log(f"Time taken: {elapsed_ms:.0f} ms")
        if not results:
            log("  No matches found.")
            continue
        avg = float(np.mean([r["match_percent"] for r in results]))
        all_avg.append(avg)
        log(f"Average Top-5 similarity: {avg:.1f}%")
        for i, r in enumerate(results, 1):
            log(f"  {i}. {r['match_percent']}%  {r['title']} | {r['company']} | {r['location']}")

    if all_avg:
        log(f"\nAverage of the per-profile Top-5 averages: {np.mean(all_avg):.1f}%")

    log("\nNote: these are similarity scores, not accuracy. Measuring accuracy would need")
    log("labelled data, e.g. a list of jobs that human reviewers judged relevant per query.")

    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved to evaluation_report.txt")


if __name__ == "__main__":
    main()
