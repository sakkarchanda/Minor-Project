import joblib
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

model = joblib.load("models/harm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

categories = {}
current = None

with open("harmful_words.txt") as f:
    for line in f:
        line = line.strip().lower()
        if line.startswith("["):
            current = line[1:-1]
            categories[current] = []
        elif line:
            categories[current].append(line)

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "/live/" in url:
        return url.split("/live/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    raise ValueError("Invalid URL")

def get_text(vid):
    try:
        t = YouTubeTranscriptApi.get_transcript(vid)
        return " ".join(x['text'] for x in t), "Transcript"
    except:
        html = requests.get(f"https://www.youtube.com/watch?v={vid}").text
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text.replace("- YouTube", "")
        desc = soup.find("meta", {"name": "description"})
        return title + " " + (desc["content"] if desc else ""), "Title + Description"

def keyword_score(text):
    found = {}
    score = 0
    t = text.lower()
    for cat, words in categories.items():
        hits = [w for w in words if w in t]
        if hits:
            found[cat] = hits
            score += 0.15 * len(hits)
    return found, min(score, 1.0)

url = input("Enter YouTube video URL: ")
vid = extract_video_id(url)

text, source = get_text(vid)

X = vectorizer.transform([text])
prob = model.predict_proba(X)[0][1]

found, k_score = keyword_score(text)
final_score = min(prob * 0.7 + k_score * 0.3, 1.0)

if final_score < 0.3:
    label = "NOT HARMFUL"
elif final_score < 0.6:
    label = "MODERATE"
else:
    label = "HARMFUL"

print("\n========== RESULT ==========")
print("Text Source:", source)
print("Final Classification:", label)
print("Risk Score:", round(final_score, 2))
print("Detected Categories:", list(found.keys()))
print("Trigger Words:", found)
