import numpy as np
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="Graduation Project Recommendation API")

# ==============================
# Models
# ==============================
class ProjectItem(BaseModel):
    id: str
    abstract: str

class ProjectRequest(BaseModel):
    problem: str
    projects: List[ProjectItem]


# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()


# ==============================
# Vectorizer
# ==============================
def get_vectorizer():
    return TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_features=10000,
        sublinear_tf=True
    )


# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}


@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"error": "Problem text is empty"}

    # 🔹 تنظيف النص
    user_text = clean_text(request.problem)

    project_texts = [
        clean_text(p.abstract) for p in request.projects
    ]

    # 🔹 Vectorization
    vectorizer = get_vectorizer()

    project_matrix = vectorizer.fit_transform(project_texts)
    user_vec = vectorizer.transform([user_text])

    # 🔹 Similarity
    similarities = cosine_similarity(user_vec, project_matrix)[0]

    # 🔹 ترتيب النتائج
    sorted_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in sorted_indices:
        score = float(similarities[idx])

        # تجاهل القيم الصغيرة جدًا
        if score < 0.1:
            continue

        results.append({
            "id": request.projects[idx].id,
            "similarity": round(score * 100, 2)
        })

    return {
        "results": results
    }