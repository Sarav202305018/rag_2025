import faiss
import pickle
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import streamlit as st  # For UI
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load & Preprocess Financial Data
def load_financial_data(file_path='/financial_data.csv'):
    """
    Load and preprocess financial data:
    - Strips column names of spaces
    - Removes NaN values
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    return df

def preprocess_text(text):
    """Clean text by removing special characters and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9.,\s]', '', text)
    return text.lower()

def chunk_text(text, chunk_size=500):
    """Split text into chunks of a given size."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 2: Create Corpus & Metadata
def build_corpus(df):
    """
    Convert structured financial data into a text-based corpus for retrieval.
    - Ensures unique Company-Year entries.
    - Constructs metadata for each entry.
    """
    corpus = []
    metadata = []

    unique_entries = df.groupby(["Company", "Year"]).first().reset_index()

    for _, row in unique_entries.iterrows():
        text_data = (
            f"Year: {row['Year']}, Company: {row['Company']}, "
            f"Category: {row['Category']}, Market Cap: {row['Market Cap(in B USD)']}B, "
            f"Revenue: {row['Revenue']}, Net Income: {row['Net Income']}, "
            f"EBITDA: {row['EBITDA']}, Debt/Equity Ratio: {row['Debt/Equity Ratio']}, "
            f"ROE: {row['ROE']}"
        )

        corpus.append(text_data)
        metadata.append(text_data)

    return corpus, metadata

# Step 3: Initialize Model & Create Indexes
# Load financial data
df = load_financial_data('/financial_data.csv')
corpus, metadata = build_corpus(df)

# Initialize pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(corpus, convert_to_numpy=True)

# Create FAISS index for vector search
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Create BM25 Index for keyword-based retrieval
tokenized_corpus = [text.lower().split() for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Step 4: Implement Multi-Stage Retrieval
def retrieve_documents(query, top_k=5):
    """
    Multi-stage retrieval using BM25 and FAISS:
    - Stage 1: BM25 gets top keyword-matching documents.
    - Stage 2: FAISS refines results using semantic similarity.
    - Confidence scores are normalized for better interpretation.
    """

    # BM25 Retrieval (Keyword-based)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    # FAISS Retrieval (Semantic-based)
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, faiss_indices = index.search(query_embedding, top_k)

    # Combine results
    combined_indices = list(set(bm25_top_indices).union(set(faiss_indices[0])))
    
    results = []
    for idx in combined_indices:
        text = metadata[idx]
        confidence = bm25_scores[idx] + np.dot(embeddings[idx], query_embedding.T)[0]
        results.append((text, confidence))

    # Normalize confidence scores (0 to 1)
    scores = np.array([r[1] for r in results]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores).flatten()

    return [(results[i][0], round(scores[i], 2)) for i in range(len(results))]

# Step 5: Implement Guardrails
def validate_query(query):
    """
    Basic query guardrails:
    - Rejects queries with harmful content.
    - Ensures financial relevance.
    """
    forbidden_words = ["hack", "attack", "illegal", "terror", "drugs"]
    if any(word in query.lower() for word in forbidden_words):
        return False, "Your query is blocked due to security reasons."
    
    if len(query.split()) < 2:  # Ensure queries are meaningful
        return False, "Query too short, please provide more context."
    
    return True, ""

# Step 6: Streamlit UI
st.title("Financial Data RAG Search")

query = st.text_input("Enter your financial query:")
if query:
    valid, message = validate_query(query)
    if not valid:
        st.error(message)
    else:
        results = retrieve_documents(query)
        if results:
            for text, confidence in results:
                st.write(f"🔹 {text} (Confidence: {confidence})")
        else:
            st.warning("No relevant data found.")
