import os
import joblib
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# ==========================================
# Add Labels
# Fake = 0
# Real = 1
# ==========================================

fake["label"] = 0
true["label"] = 1

# ==========================================
# Combine Dataset
# ==========================================

data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = shuffle(data, random_state=42)

# Remove missing values
data = data.fillna("")

# ==========================================
# Combine Title + Text
# ==========================================

data["content"] = data["title"] + " " + data["text"]

# Keep required columns
data = data[["content", "label"]]

print("Total Articles :", len(data))
print()

# ==========================================
# Split Dataset
# ==========================================

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print()

# ==========================================
# TF-IDF Vectorizer
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ==========================================
# Train Model
# ==========================================

print("Training AI Model...")

model = LogisticRegression(
    C=2.0,
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Evaluate Model
# ==========================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("=" * 60)
print("MODEL ACCURACY :", round(accuracy * 100, 2), "%")
print("=" * 60)

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, prediction))

# ==========================================
# Test Sample News
# ==========================================

print("\nTesting Sample News...\n")

sample_news = [

    "The Indian Space Research Organisation successfully launched a PSLV rocket carrying satellites from Sriharikota.",

    "Scientists confirmed dinosaurs are alive in India and are working inside banks."

]

sample_vector = vectorizer.transform(sample_news)

sample_prediction = model.predict(sample_vector)
sample_probability = model.predict_proba(sample_vector)

for i in range(len(sample_news)):

    print("-" * 60)
    print("NEWS:")
    print(sample_news[i])
    print()

    if sample_prediction[i] == 1:
        print("Prediction : REAL NEWS")
    else:
        print("Prediction : FAKE NEWS")

    print("Confidence :", round(max(sample_probability[i]) * 100, 2), "%")

# ==========================================
# Save Model
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n" + "=" * 60)
print("✅ Model trained successfully!")
print("✅ Model saved in model/fake_news_model.pkl")
print("✅ Vectorizer saved in model/vectorizer.pkl")
print("=" * 60)