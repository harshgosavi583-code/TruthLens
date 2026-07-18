from flask import Flask, render_template, request
import json
import os
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime

app = Flask(__name__)

# =====================================
# Load AI Model
# =====================================
try:
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    model = None
    vectorizer = None
    print("❌ Error Loading AI Model:", e)


# =====================================
# Helper Functions
# =====================================
def load_users():

    if not os.path.exists("users.json"):
        with open("users.json", "w") as f:
            json.dump([], f)

    with open("users.json", "r") as f:
        return json.load(f)


def save_users(users):

    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)


def save_history(url, prediction, confidence):

    if not os.path.exists("history.json"):
        with open("history.json", "w") as f:
            json.dump([], f)

    with open("history.json", "r") as f:
        history = json.load(f)

    history.append({
        "date": datetime.now().strftime("%d %b %Y %I:%M %p"),
        "url": url,
        "prediction": prediction,
        "confidence": confidence
    })

    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)
# =====================================
# Home
# =====================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# Register
# =====================================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            return render_template(
                "register.html",
                message="Passwords do not match!"
            )

        users = load_users()

        for user in users:
            if user["email"] == email:
                return render_template(
                    "register.html",
                    message="Email already exists!"
                )

        users.append({
            "name": name,
            "email": email,
            "password": password
        })

        save_users(users)

        return render_template("dashboard.html")

    return render_template("register.html")


# =====================================
# Login
# =====================================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        users = load_users()

        for user in users:

            if user["email"] == email and user["password"] == password:
                return render_template("dashboard.html")

        return render_template(
            "login.html",
            message="Invalid Email or Password!"
        )

    return render_template("login.html")


# =====================================
# Dashboard
# =====================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# =====================================
# Detect Page
# =====================================
@app.route("/detect")
def detect():
    return render_template("detect.html")


# =====================================
# AI Prediction
# =====================================
@app.route("/result", methods=["POST"])
def result():

    if model is None or vectorizer is None:
        return render_template(
            "result.html",
            prediction="⚠ AI Model Not Loaded",
            confidence="",
            news="",
            url=""
        )

    url = request.form.get("url", "").strip()
    trusted_sites = [
    "bbc.com",
    "indianexpress.com",
    "thehindu.com",
    "ndtv.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "reuters.com",
    "apnews.com"
]
    news = request.form.get("news", "").strip()

    # Empty URL
    if url == "" and news == "":
        return render_template(
        "result.html",
        prediction="⚠ Please enter a News URL or paste News Text.",
        confidence="",
        news="",
        url=""
        )

    # Invalid URL
    # Invalid URL
    # Check URL only if user entered one
    if url and not (
        url.startswith("http://") or
        url.startswith("https://")
        ):
        return render_template(
            "result.html",
            prediction="⚠ Invalid URL",
            confidence="",
            news="Please enter a valid URL starting with http:// or https://",
            url=url
            )
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9"
}
        if url:
            response = requests.get(
                url,
                headers=headers,
                timeout=10
                )
            response.raise_for_status()
            soup = BeautifulSoup(
                response.text,
                "html.parser"
                )
            paragraphs = soup.select(
                "article p, main p, .article-body p, .story p, p"
                )
            news = " ".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
                )
            print("\n" + "=" * 60)
            print("Extracted Article Text:")
            print(news[:2000])   # Print the first 2000 characters
            print("=" * 60)

        if not news.strip():
            return render_template(
                "result.html",
                prediction="⚠ Unable to extract enough article text.",
                confidence="",
                news="",
                url=url
            )

        # AI Prediction
        news_vector = vectorizer.transform([news])

        prediction = int(
            model.predict(news_vector)[0]
        )

        probabilities = model.predict_proba(
            news_vector
        )[0]

        fake_probability = round(
            probabilities[0] * 100,
            2
        )

        real_probability = round(
            probabilities[1] * 100,
            2
        )

        confidence = (
            f"{max(fake_probability, real_probability):.2f}%"
        )

        print("\n" + "=" * 60)
        print("URL :", url)
        print("Prediction :", prediction)
        print("Fake Probability :", fake_probability)
        print("Real Probability :", real_probability)
        print("Confidence :", confidence)
        print("=" * 60)

        if prediction == 1:
            prediction_text = "✅ Real News"
        else:
            prediction_text = "❌ Fake News"
        if any(site in url.lower() for site in trusted_sites):
            prediction_text = "✅ Real News"
            confidence = "Trusted Source"
        save_history(
            url=url,
            prediction=prediction_text,
            confidence=confidence
        )

        return render_template(
            "result.html",
            prediction=prediction_text,
            confidence=confidence,
            news=news[:1500] + "...",
            url=url
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction="⚠ Error reading URL",
            confidence="",
            news=str(e),
            url=url
        )


# =====================================
# History
# =====================================
@app.route("/history")
def history():

    if not os.path.exists("history.json"):
        with open("history.json", "w") as f:
            json.dump([], f)

    with open("history.json", "r") as f:
        history = json.load(f)

    history.reverse()   # newest first

    return render_template(
        "history.html",
        history=history
    )


# =====================================
# About
# =====================================
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/learn-more")
def learn_more():
    return render_template("about_project.html")

# =====================================
# Run Flask
# =====================================
if __name__ == "__main__":
    app.run(debug=True)