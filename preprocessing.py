"""
preprocessing.py
----------------
Shared cleaning code used by BOTH train_model.py and recommender.py.

Why is it shared?
The user's skills must be cleaned in exactly the same way as the job postings.
If the two were cleaned differently, the words would not match and the
similarity scores would be wrong.
"""

import re
import pandas as pd

# Different datasets name their columns differently.
# For each field we need, we list the names we are willing to accept.
COLUMN_ALIASES = {
    "title": ["title", "job_title", "jobtitle", "position", "job_name"],
    "description": ["description", "job_description", "jobdescription", "details", "job_details"],
    "company": ["company_name", "company", "employer", "organization", "companyname"],
    "location": ["location", "job_location", "city", "place"],
    "skills": ["skills_desc", "skills", "required_skills", "skill", "key_skills"],
    "url": ["job_posting_url", "job_url", "url", "link", "job_link", "application_url"],
}

# Descriptions are shortened before saving so the app uses less memory.
MAX_SAVED_DESCRIPTION_CHARS = 1000


def clean_text(text):
    """Lowercase the text, remove HTML tags/punctuation and extra spaces.

    We keep '+' and '#' so skills like 'c++' and 'c#' survive.
    """
    if not isinstance(text, str):
        return ""                                    # missing values (NaN) -> empty text
    text = text.lower()                              # 'Python' and 'python' become the same word
    text = re.sub(r"<[^>]+>", " ", text)             # remove HTML tags such as <br>
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)       # replace punctuation/symbols with a space
    text = re.sub(r"\s+", " ", text).strip()         # collapse repeated whitespace
    return text


def find_column(df, field):
    """Return the real column name in df that matches one of the aliases, or None."""
    for name in COLUMN_ALIASES[field]:
        if name in df.columns:
            return name
    return None


def read_csv_safely(path):
    """Read the CSV; retry with a different encoding if UTF-8 fails."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def load_and_clean(csv_path):
    """Load the CSV and return (clean_dataframe, stats_dictionary).

    clean_dataframe columns: title, company, location, description, skills, url, search_text
    """
    df = read_csv_safely(csv_path)
    stats = {"rows_loaded": len(df)}

    # Make column names predictable: lowercase, spaces -> underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print("Columns found in the CSV:", list(df.columns))

    # Map the dataset's columns to our standard names
    mapping = {field: find_column(df, field) for field in COLUMN_ALIASES}
    print("Column mapping used:", mapping)
    if mapping["title"] is None and mapping["description"] is None:
        raise ValueError(
            "Could not find a job title or job description column. "
            "Columns in your file: " + ", ".join(df.columns)
        )

    clean = pd.DataFrame()
    for field, column in mapping.items():
        clean[field] = df[column] if column is not None else ""

    # 1. Inspect missing values (before filling them)
    print("\nMissing values per field:")
    print(clean.isna().sum().to_string())

    # 2. Handle missing values: replace with empty text and make everything a string
    clean = clean.fillna("").astype(str)
    for field in clean.columns:
        clean[field] = clean[field].str.strip()

    # 3. Remove duplicate postings (same title + company + location + description)
    before = len(clean)
    clean = clean.drop_duplicates(subset=["title", "company", "location", "description"])
    stats["duplicates_removed"] = before - len(clean)

    # 4. Combine title + skills + description into one searchable text, then clean it
    combined = clean["title"] + " " + clean["skills"] + " " + clean["description"]
    clean["search_text"] = combined.map(clean_text)

    # 5. Drop rows that have no usable text at all
    before = len(clean)
    clean = clean[clean["search_text"] != ""].reset_index(drop=True)
    stats["empty_text_removed"] = before - len(clean)

    stats["non_empty_descriptions"] = int((clean["description"] != "").sum())
    stats["jobs_final"] = len(clean)

    # Keep only a short description for display (saves memory in the web app)
    clean["description"] = clean["description"].str.slice(0, MAX_SAVED_DESCRIPTION_CHARS)
    return clean, stats
