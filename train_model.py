"""
train_model.py
--------------
Run this ONCE (and again whenever you change the dataset).

It: loads data/jobs.csv -> cleans it -> trains TF-IDF -> converts every job to a
vector -> saves everything into the model/ folder.

Run it with:
    python3 train_model.py
Optional (use another CSV path):
    python3 train_model.py path/to/other.csv
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import load_and_clean

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "jobs.csv"
MODEL_DIR = BASE_DIR / "model"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: dataset not found at {csv_path}")
        print("Put your CSV at data/jobs.csv and run this again.")
        sys.exit(1)

    start = time.perf_counter()

    # Steps 1-5: load and clean
    print(f"Loading {csv_path} ...")
    jobs, stats = load_and_clean(csv_path)
    print("\nCleaning summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Step 6: train TF-IDF
    # - stop_words: ignore very common English words like 'the', 'and'
    # - min_df=2: ignore words that appear in only one job (usually typos/noise)
    # - max_df=0.8: ignore words that appear in more than 80% of jobs (not useful)
    # - max_features: keep the 50,000 most useful words (keeps the model small)
    # - sublinear_tf: dampens the effect of a word repeated many times
    # - dtype float32: half the memory of the default
    print("\nTraining TF-IDF (this can take a few minutes on a big dataset)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"[a-z0-9][a-z0-9+#]*",
        min_df=2,
        max_df=0.8,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    job_vectors = vectorizer.fit_transform(jobs["search_text"])

    print(f"  Job matrix shape (jobs x vocabulary): {job_vectors.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Step 7: save. search_text is not needed later, so we drop it to save space.
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl", compress=3)
    joblib.dump(job_vectors, MODEL_DIR / "job_vectors.pkl", compress=3)
    joblib.dump(jobs.drop(columns=["search_text"]), MODEL_DIR / "jobs_clean.pkl", compress=3)

    print(f"\nSaved 3 files in {MODEL_DIR}:")
    for f in sorted(MODEL_DIR.glob("*.pkl")):
        print(f"  {f.name}  ({f.stat().st_size / 1_000_000:.1f} MB)")
    print(f"\nDone in {time.perf_counter() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
