from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),   # دقة أعلى
        max_features=10000,
        sublinear_tf=True
    )

def create_embeddings(texts, vectorizer):
    return vectorizer.fit_transform(texts)