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
    description: str   # ✅ بدل abstract

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
@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"results": []}

    user_text = clean_text(request.problem)

    # ✅ فلترة الداتا
    valid_projects = [
        p for p in request.projects
        if p.description and p.description.strip() != ""
    ]

    if not valid_projects:
        return {"results": []}

    project_texts = [
        clean_text(p.description) for p in valid_projects
    ]

    # 🔥 أهم fix
    all_texts = project_texts + [user_text]

    vectorizer = get_vectorizer()
    matrix = vectorizer.fit_transform(all_texts)

    project_matrix = matrix[:-1]
    user_vec = matrix[-1]

    similarities = cosine_similarity(user_vec, project_matrix)[0]

    sorted_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in sorted_indices:
        score = float(similarities[idx])

        if score < 0.05:
            continue

        results.append({
            "id": valid_projects[idx].id,
            "similarity": round(score * 100, 2)
        })

    return {
        "results": results
    }