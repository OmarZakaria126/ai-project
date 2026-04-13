import numpy as np
from config import DUPLICATION_THRESHOLD

def check_duplication(idea_embedding, previous_embeddings, previous_ideas):

    similarities = (idea_embedding @ previous_embeddings.T)[0]

    duplicates = []

    for i, score in enumerate(similarities):
        if score >= DUPLICATION_THRESHOLD:
            duplicates.append({
                "idea": previous_ideas[i],
                "similarity_percentage": round(float(score) * 100, 2)
            })

    return duplicates
