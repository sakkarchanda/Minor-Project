import pandas as pd
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

nltk.download("stopwords")

df = pd.read_csv("Harmful _full_agreement.csv")

harmful = df['transcript'].dropna().sample(5000, random_state=42)
harmful = pd.DataFrame({"text": harmful, "label": 1})

safe = pd.Series([
    "this video discusses current events and opinions",
    "people share thoughts and comments online",
    "public discussion on social media platforms",
    "news reporting and analysis video",
    "interview with public figures",
    "this video talks about social issues",
    "general conversation and debate",
    "discussion about policies and society",
    "viewer comments and reactions",
    "opinion based discussion video",
    "this channel posts regular content",
    "general information video",
    "this video explains different viewpoints",
    "people expressing their opinions",
    "discussion about trends and topics"
] * 350)

safe = pd.DataFrame({"text": safe, "label": 0})

data = pd.concat([harmful, safe], ignore_index=True)

stop_words = set(stopwords.words("english"))

data['text'] = (
    data['text']
    .astype(str)
    .str.lower()
    .str.split()
    .apply(lambda x: " ".join(w for w in x if w.isalpha() and w not in stop_words))
)

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 2),
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=3000, C=0.7)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

joblib.dump(model, "models/harm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
