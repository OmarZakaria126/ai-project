import numpy as np
from config import RECOMMEND_THRESHOLD, TOP_K_RECOMMEND

def recommend(user_embedding, external_embeddings, external_projects):

    similarities = (user_embedding @ external_embeddings.T)[0]

    if max(similarities) < RECOMMEND_THRESHOLD:
        return None

    top_indices = np.argsort(similarities)[::-1][:TOP_K_RECOMMEND]

    results = []
    for idx in top_indices:
        proj = external_projects[idx]
        results.append({
            "project_name": proj["Project Name"],
            "year": proj["Year"],
            "specialization": proj["Specialization"],
            "tools": proj["Tools"],
            "description": proj["Introduction"]
        })

    return results