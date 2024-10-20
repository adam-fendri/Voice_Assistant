import os
import torch
import uuid
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'financial-rag-index'
dimension = 768  
metric = 'cosine'  

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

tokenizer = AutoTokenizer.from_pretrained("sujet-ai/Marsilia-Embeddings-FR-Base")
model = AutoModel.from_pretrained("sujet-ai/Marsilia-Embeddings-FR-Base")

def embed(text):
    """Embeds the input text with the Marsilia model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def chunk_text(text, max_length=512, overlap=50):
    """Splits the text into chunks for embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

dataset = load_dataset('sujet-ai/Sujet-Financial-RAG-FR-Dataset', split='train').select(range(150))  # Limiting to first 150 entries for example

batch = []
batch_size = 100  

for item in dataset:
    query = item["question"]
    context = item["context"]
    context_chunks = chunk_text(context)

    for chunk in context_chunks:
        combined_text = query + " " + chunk
        combined_embed = embed(combined_text)
        unique_id = str(uuid.uuid4())  
        metadata = {"text": combined_text}
        batch.append((unique_id, combined_embed, metadata))

        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            batch = []  

if batch:
    index.upsert(vectors=batch)  

print("Data has been uploaded to Pinecone index.")
