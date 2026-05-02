from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(projects, model):
    return model.encode(
        [f"{p['Project Name']} {p['Introduction']}" for p in projects]
    )

def get_similarity(vector, matrix):
    return cosine_similarity(vector, matrix)[0]