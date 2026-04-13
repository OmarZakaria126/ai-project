import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI(title="Graduation Project Recommendation API")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RECOMMEND_THRESHOLD = 0.55
DUPLICATION_THRESHOLD = 0.6
TOP_K_RECOMMEND = 3


class ProjectRequest(BaseModel):
    problem: str
    previousIdeas: list[str]


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}


@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"error": "Problem text is empty"}

    user_embedding = np.array(get_embedding(request.problem)).reshape(1, -1)

    if not request.previousIdeas:
        return {
            "recommendations": [],
            "status": "accepted",
            "duplicates": []
        }

    previous_embeddings = np.array(
        [get_embedding(idea) for idea in request.previousIdeas]
    )

    similarities = cosine_similarity(
        user_embedding,
        previous_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:TOP_K_RECOMMEND]

    recommendations = []
    duplicates = []

    for idx in top_indices:
        score = float(similarities[idx])
        idea_text = request.previousIdeas[idx]

        item = {
            "idea": idea_text,
            "similarity_percentage": round(score * 100, 2)
        }

        recommendations.append(item)

        if score >= DUPLICATION_THRESHOLD:
            duplicates.append(item)

    status = "rejected" if duplicates else "accepted"

    return {
        "recommendations": recommendations,
        "status": status,
        "duplicates": duplicates
    }