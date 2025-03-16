import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# Load financial data
def load_financial_data(file_path='/content/sample_data/financial_data.csv'):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names
    df = df.dropna()
    return df

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,\s]', '', text)
    return text.lower()

# Chunk text into smaller parts
def chunk_text(text, chunk_size=500):  # Increased chunk size to reduce excessive splitting
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Create text corpus and metadata
def build_corpus(df):
    corpus = []
    metadata = []

    # 🔹 Ensure only ONE entry per Company-Year
    unique_entries = df.groupby(["Company", "Year"]).first().reset_index()

    for _, row in unique_entries.iterrows():  # Use only unique Company-Year rows
        text_data = (
            f"Year: {row['Year']}, Company: {row['Company']}, "
            f"Category: {row['Category']}, Market Cap: {row['Market Cap(in B USD)']}B, "
            f"Revenue: {row['Revenue']}, Net Income: {row['Net Income']}, "
            f"EBITDA: {row['EBITDA']}, Debt/Equity Ratio: {row['Debt/Equity Ratio']}, "
            f"ROE: {row['ROE']}"
        )

        corpus.append(text_data)  # Store only 1 entry per Company-Year
        metadata.append(text_data)  # Keep metadata consistent

    return corpus, metadata


# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess data
df = load_financial_data('/content/sample_data/financial_data.csv')
corpus, metadata = build_corpus(df)

# 🔹 Debug: Print corpus and metadata length
print(f"📌 Corpus Length: {len(corpus)}")  # Should match metadata
print(f"📌 Metadata Length: {len(metadata)}")  # Should match your friend's

# Generate embeddings
embeddings = model.encode(corpus, convert_to_numpy=True)
# 🔹 Debug: Print FAISS embedding shape
print(f"✅ FAISS Embeddings Shape: {embeddings.shape}")  # Should be (13, embedding_dim)


# Create FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
# 🔹 Debug: Print FAISS index size before saving
print(f"✅ FAISS Index Size Before Saving: {index.ntotal}")  # Should match corpus length

# Create BM25 Index
tokenized_corpus = [text.lower().split() for text in corpus]  # Ensure consistent tokenization
bm25 = BM25Okapi(tokenized_corpus)

# Save index and metadata
faiss.write_index(index, "vector_index.faiss")
# Debug final FAISS index size
print(f"✅ Final FAISS Index Size: {index.ntotal}")  # Should match metadata length

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
with open("bm25_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)  # Ensure corpus is saved

print("✅ FAISS index, metadata, and BM25 index successfully created and saved.")

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
