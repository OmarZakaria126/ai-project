import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import create_embeddings, transform_input

app = FastAPI()

class ProjectRequest(BaseModel):
    problem: str
    previousIdeas: list[str]

@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}

@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"error": "Problem text is empty"}

    previous = request.previousIdeas[:30]

    if not previous:
        return {
            "recommendations": [],
            "status": "accepted",
            "duplicates": []
        }

    matrix = create_embeddings(previous)
    user_vec = transform_input(request.problem)

    similarities = cosine_similarity(user_vec, matrix)[0]

    top_indices = np.argsort(similarities)[::-1][:5]

    recommendations = []
    duplicates = []

    for idx in top_indices:
        score = float(similarities[idx])

        item = {
            "idea": previous[idx],
            "similarity_percentage": round(score * 100, 2)
        }

        recommendations.append(item)

        if score >= 0.65:  # 👈 رفعنا threshold للدقة
            duplicates.append(item)

    return {
        "recommendations": recommendations,
        "status": "rejected" if duplicates else "accepted",
        "duplicates": duplicates
    }