import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI(title="Graduation Project Recommendation API")

# API Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
RECOMMEND_THRESHOLD = 0.55
DUPLICATION_THRESHOLD = 0.6
TOP_K_RECOMMEND = 3


class ProjectRequest(BaseModel):
    problem: str
    previousIdeas: list[str]


# ✅ Safe embedding function
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print("Embedding Error:", str(e))
        return None


@app.get("/")
def home():
    return {"message": "AI service is running 🚀"}


@app.post("/check")
def check_duplication(request: ProjectRequest):

    if not request.problem:
        return {"error": "Problem text is empty"}

    # ✅ user embedding
    user_embedding = get_embedding(request.problem)
    if user_embedding is None:
        return {"error": "Failed to process problem text"}

    user_embedding = np.array(user_embedding).reshape(1, -1)

    # ✅ لو مفيش أفكار سابقة
    if not request.previousIdeas:
        return {
            "recommendations": [],
            "status": "accepted",
            "duplicates": []
        }

    # ✅ previous embeddings
    previous_embeddings_list = []
    valid_ideas = []

    for idea in request.previousIdeas:
        emb = get_embedding(idea)
        if emb is not None:
            previous_embeddings_list.append(emb)
            valid_ideas.append(idea)

    if not previous_embeddings_list:
        return {"error": "Failed to process previous ideas"}

    previous_embeddings = np.array(previous_embeddings_list)

    # ✅ similarity
    similarities = cosine_similarity(
        user_embedding,
        previous_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:TOP_K_RECOMMEND]

    recommendations = []
    duplicates = []

    for idx in top_indices:
        score = float(similarities[idx])
        idea_text = valid_ideas[idx]

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