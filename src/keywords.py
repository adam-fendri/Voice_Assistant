from datasets import load_dataset
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_dynamic_keywords():
    
    dataset = load_dataset("sujet-ai/Sujet-Financial-RAG-FR-Dataset", split="train[:200]")
    # Extract and process question keywords
    all_questions = " ".join(dataset['question'])
    question_tokens = word_tokenize(all_questions)
    filtered_questions = [word for word in question_tokens if word.lower() not in stopwords.words('french')]
    question_counts = Counter(filtered_questions)
    common_question_keywords = [word for word, _ in question_counts.most_common(100)]

    # Extract and process context keywords
    all_context = " ".join(dataset['context'])
    context_tokens = word_tokenize(all_context)
    filtered_context = [word for word in context_tokens if word.lower() not in stopwords.words('french')]
    context_counts = Counter(filtered_context)
    common_context_keywords = [word for word, _ in context_counts.most_common(100)]

    dynamic_keywords = list(set(common_question_keywords + common_context_keywords))

    return dynamic_keywords
