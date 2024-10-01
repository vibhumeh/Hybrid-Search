from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import scipy
import pandas as pd

# Load pretrained embedding model for dense retrieval
dense_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example textbook sections for retrieval
df = pd.read_csv('./data_clean/textbooks/paragraphed/paragraphs.csv')

# Extract the 'Paragraph' column as the documents
documents = df['Paragraph'].tolist()

# Sparse Retrieval: Use TF-IDF
vectorizer = TfidfVectorizer()
sparse_embeddings = vectorizer.fit_transform(documents)

# Dense Retrieval: Convert documents into dense embeddings
dense_embeddings = dense_model.encode(documents, show_progress_bar=True)

# Create FAISS index for dense retrieval (we'll use cosine similarity)
dense_index = faiss.IndexFlatL2(dense_embeddings.shape[1])
dense_index.add(np.array(dense_embeddings).astype(np.float32))
def hybrid_search(query, alpha=0.5, top_k=3):
    # 1. Sparse retrieval (TF-IDF)
    sparse_query_embedding = vectorizer.transform([query])
    sparse_scores = np.dot(sparse_query_embedding, sparse_embeddings.T).toarray()[0]

    # 2. Dense retrieval (Embeddings)
    dense_query_embedding = dense_model.encode([query])
    _, dense_indices = dense_index.search(np.array(dense_query_embedding).astype(np.float32), top_k)
    dense_scores = 1 - np.linalg.norm(dense_embeddings[dense_indices[0]] - dense_query_embedding, axis=1)

    # 3. Combine scores with alpha
    normalized_sparse_scores = sparse_scores / np.max(sparse_scores)
    normalized_dense_scores = dense_scores / np.max(dense_scores)

    # Hybrid score = alpha * dense + (1 - alpha) * sparse
    hybrid_scores = alpha * normalized_dense_scores + (1 - alpha) * normalized_sparse_scores[:top_k]

    # Rank the documents based on hybrid scores
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    
    return [(documents[idx], hybrid_scores[idx]) for idx in ranked_indices]
sparse_embeddings_file = "./sparse_embeddings.npz"
scipy.sparse.save_npz(sparse_embeddings_file, sparse_embeddings)

# Save Dense Embeddings
dense_embeddings_file = "./dense_embeddings.npy"
np.save(dense_embeddings_file, dense_embeddings)

# Save FAISS Index
faiss_index_file = "./dense_index.faiss"
faiss.write_index(dense_index, faiss_index_file)
# Example usage:
query = "What is hypertension?"
results = hybrid_search(query, alpha=0.7)
for doc, score in results:
    print(f"Score: {score:.4f} - Document: {doc}")
