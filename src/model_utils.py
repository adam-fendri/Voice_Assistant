from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("sujet-ai/Marsilia-Embeddings-FR-Base")
model = AutoModel.from_pretrained("sujet-ai/Marsilia-Embeddings-FR-Base")

def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze()
    return vector.detach().cpu().numpy()

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros(vector.shape).tolist()  # Avoid division by zero
    normalized_vector = vector / norm
    return np.clip(normalized_vector, -1.0, 1.0).tolist()
