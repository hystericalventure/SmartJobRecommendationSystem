"""
recommender.py
--------------
The recommendation logic. It knows nothing about Flask, so it can be tested on its own.

Main function:
    recommend_jobs(user_skills, top_n=5)
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "model"
MIN_SCORE = 0.01          # ignore jobs with (almost) zero similarity
MAX_INPUT_CHARS = 500
SHORT_DESCRIPTION_CHARS = 300

_cache = {}               # loaded model files are kept here so we read the disk only once


class InvalidInputError(Exception):
    """The user's input is empty, too short or too long."""


class ModelNotFoundError(Exception):
    """The trained model files are missing (run train_model.py first)."""


def load_artifacts():
    """Load the vectorizer, job vectors and job table from model/ (only the first time)."""
    if _cache:
        return _cache

    files = {
        "vectorizer": MODEL_DIR / "tfidf_vectorizer.pkl",
        "job_vectors": MODEL_DIR / "job_vectors.pkl",
        "jobs": MODEL_DIR / "jobs_clean.pkl",
    }
    missing = [path.name for path in files.values() if not path.exists()]
    if missing:
        raise ModelNotFoundError(
            "Model files not found: " + ", ".join(missing) + ". Run 'python3 train_model.py' first."
        )

    for name, path in files.items():
        _cache[name] = joblib.load(path)
    _cache["feature_names"] = _cache["vectorizer"].get_feature_names_out()
    return _cache


def _short_text(text, limit=SHORT_DESCRIPTION_CHARS):
    text = " ".join(str(text).split())               # remove line breaks / extra spaces
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def recommend_jobs(user_skills, top_n=5):
    """Return the top_n jobs most similar to the user's skills.

    Returns a list of dictionaries (can be empty if nothing matches).
    Raises InvalidInputError or ModelNotFoundError for problems.
    """
    # 1. Validate the input
    if not isinstance(user_skills, str) or not user_skills.strip():
        raise InvalidInputError("Please enter at least one skill.")
    if len(user_skills) > MAX_INPUT_CHARS:
        raise InvalidInputError(f"Input is too long. Please use at most {MAX_INPUT_CHARS} characters.")
    cleaned = clean_text(user_skills)
    if len(cleaned) < 2:
        raise InvalidInputError("Input is too short. Enter real skills, e.g. 'Python, SQL'.")

    # 2. Load the model and turn the user's skills into a TF-IDF vector
    art = load_artifacts()
    user_vec = art["vectorizer"].transform([cleaned])
    if user_vec.nnz == 0:
        return []                                    # none of the words exist in our vocabulary

    # 3. Cosine similarity between the user vector and ALL job vectors
    scores = cosine_similarity(user_vec, art["job_vectors"]).ravel()

    # 4. Pick the top_n highest scores and sort them (best first)
    n = min(top_n, len(scores))
    top_idx = np.argpartition(-scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    user_weights = dict(zip(user_vec.indices, user_vec.data))
    feature_names = art["feature_names"]
    jobs = art["jobs"]

    results = []
    for idx in top_idx:
        score = float(scores[idx])
        if score < MIN_SCORE:
            continue

        # Which of the user's words also appear in this job? (real overlap, not invented)
        job_terms = set(art["job_vectors"][idx].indices)
        common = [i for i in user_weights if i in job_terms]
        common.sort(key=lambda i: -user_weights[i])

        row = jobs.iloc[idx]
        url = row["url"] if str(row["url"]).startswith("http") else ""
        results.append({
            "title": row["title"] or "Untitled job",
            "company": row["company"],
            "location": row["location"],
            "description": _short_text(row["description"]),
            "skills": _short_text(row["skills"], 150),
            "matched_terms": [feature_names[i] for i in common][:8],
            "url": url,
            "score": round(score, 4),
            "match_percent": round(score * 100, 1),   # similarity score, NOT a hiring probability
        })
    return results


if __name__ == "__main__":
    # Quick manual test:  python3 recommender.py
    for job in recommend_jobs("Python, Machine Learning, SQL"):
        print(f"{job['match_percent']}%  {job['title']}  |  {job['company']}  |  {job['location']}")
