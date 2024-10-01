
import scipy
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
file=open("vectorizer.pkl",'rb')
documents = pd.read_csv('./data_clean/textbooks/paragraphed/paragraphs.csv')["Paragraph"]
vectorizer = pickle.load(file)
sparse_embeddings = scipy.sparse.load_npz("./sparse_embeddings.npz")

# Load Dense Embeddings
dense_embeddings = np.load("./dense_embeddings.npy")
dense_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("-------------------------------------------------------------------------------")
# Load FAISS Index
dense_index = faiss.read_index("./dense_index.faiss")

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


###
def answer_multiple_choice_question(question, options, alpha=0.5, top_k=6):
    # Step 1: Use the question as a query for hybrid search
    retrieved_docs = hybrid_search(question, alpha=alpha, top_k=top_k)
    
    # Step 2: For each option, calculate its similarity to the retrieved documents
    option_scores = []
    for option in options:
        option_embedding = dense_model.encode([option])  # Get dense embedding for the option
        similarity_scores = []

        for doc, _ in retrieved_docs:  # For each retrieved document
            doc_embedding = dense_model.encode([doc])
            similarity = 1 - np.linalg.norm(option_embedding - doc_embedding)  # Calculate similarity (cosine or L2)
            similarity_scores.append(similarity)
        
        # Average similarity score for the option
        avg_similarity = np.mean(similarity_scores)
        option_scores.append((option, avg_similarity))

    # Step 3: Choose the option with the highest average similarity score
    best_option = max(option_scores, key=lambda x: x[1])
    0
    return option_scores #best_option

def answer_multiple_choice_question2(question, options, alpha=0.7, top_k=3):
    # Step 1: Use the question as a query for hybrid search
    retrieved_docs = hybrid_search(question, alpha=alpha, top_k=top_k)
    
    # Step 2: For each option, calculate its similarity to the retrieved documents
    option_scores = []
    for option_key, option_text in options.items():
        # Remove the label (A:, B:, etc.) from the option
        option_text_cleaned = option_text.split(":")[-1].strip()

        option_embedding = dense_model.encode([option_text_cleaned])  # Get dense embedding for the cleaned option
        similarity_scores = []

        for doc, _ in retrieved_docs:  # For each retrieved document
            doc_embedding = dense_model.encode([doc])
            similarity = 1 - np.linalg.norm(option_embedding - doc_embedding)  # Calculate similarity
            similarity_scores.append(similarity)
        
        # Average similarity score for the option
        avg_similarity = np.mean(similarity_scores)
        option_scores.append((option_key, avg_similarity))

    # Step 3: Choose the option with the highest average similarity score
    best_option = max(option_scores, key=lambda x: x[1])
    
    return best_option[0]  # Return the option key (A, B, C, etc.)


correct_answers=0
total_questions=0
with open('./data_clean/questions/US/US_qbank.jsonl', 'r') as f:
    for line in f:
        # Parse each line as a JSON object
        data = json.loads(line)
        
        # Extract the question, options, and correct answer
        question = data["question"]
        options = data["options"]
        correct_answer = data["answer"]
        # Run the question through the answer function
        predicted_answer = answer_multiple_choice_question2(question, options, alpha=0.7)
        # Compare the predicted answer with the correct answer
        
        if predicted_answer == correct_answer:
            correct_answers += 1 
        total_questions += 1

# Output the number of correct answers
print(f"Total questions: {total_questions}")
print(f"Correct answers: {correct_answers}")
print(f"Accuracy: {correct_answers / total_questions * 100:.2f}%")
# Example usage
question = "What is the primary requirement for cells in the human body to survive?"
options = [
           "Continuous supply of oxygen and carbon dioxide exchange..",
           "Coordinated interaction between cells and organs for daily activities.",
           "The ability to maintain a constant body temperature.",
           "Adequate cellular energy supplies, maintenance of the intracellular environment, and defense against external threats.",]

best_option = answer_multiple_choice_question(question, options, alpha=0.7)
question = "A 4670-g (10-lb 5-oz) male newborn is delivered at term to a 26-year-old woman after prolonged labor. Apgar scores are 9 and 9 at 1 and 5 minutes. Examination in the delivery room shows swelling, tenderness, and crepitus over the left clavicle. There is decreased movement of the left upper extremity. Movement of the hands and wrists are normal. A grasping reflex is normal in both hands. An asymmetric Moro reflex is present. The remainder of the examination shows no abnormalities and an anteroposterior x-ray confirms the diagnosis. Which of the following is the most appropriate next step in management?"

options = [
    "Nerve conduction study",
    "Surgical fixation",
    "Physical therapy",
    "Pin sleeve to the shirt",
    "Splinting of the arm",
    "MRI of the clavicle"
]

best_option = answer_multiple_choice_question(question, options, alpha=0.7)

print(best_option)
print(f"The best answer is: {best_option[0]} with a score of {best_option[1]:.4f}")
###

""" query = "What is hypertension?"
results = hybrid_search(query, alpha=0.7)
for doc, score in results:
    print(f"Score: {score:.4f} - Document: {doc}")
file.close() """