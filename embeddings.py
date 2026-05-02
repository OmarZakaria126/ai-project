from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),   # 👈 مهم جدًا للدقة
    max_features=5000
)

def create_embeddings(texts):
    cleaned = [clean_text(t) for t in texts]
    return vectorizer.fit_transform(cleaned)

def transform_input(text):
    return vectorizer.transform([clean_text(text)])