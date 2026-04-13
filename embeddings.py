from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(projects):
    return model.encode(
        [f"{p['Project Name']} {p['Introduction']}" for p in projects]
    )

def get_similarity(vector, matrix):
    return cosine_similarity(vector, matrix)[0]
