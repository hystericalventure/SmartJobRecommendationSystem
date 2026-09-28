"""
app.py
------
The Flask web application. It only handles web pages;
all recommendation logic lives in recommender.py.

Run it with:
    python3 app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, redirect, render_template, request, url_for

from recommender import InvalidInputError, ModelNotFoundError, recommend_jobs

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", skills="", jobs=None, error=None)


@app.route("/recommend", methods=["POST"])
def recommend():
    skills = request.form.get("skills", "")
    jobs, error = None, None

    try:
        jobs = recommend_jobs(skills, top_n=5)
        if not jobs:
            error = "No meaningful matches found. Try more common skills, e.g. 'Python, SQL, Excel'."
    except InvalidInputError as e:
        error = str(e)
    except ModelNotFoundError as e:
        error = str(e)
    except Exception as e:                           # any other problem (e.g. corrupted data)
        app.logger.exception("Unexpected error")
        error = "Something went wrong while reading the job data. Please retrain the model and try again."

    return render_template("index.html", skills=skills, jobs=jobs, error=error)


@app.route("/recommend", methods=["GET"])
def recommend_redirect():
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
