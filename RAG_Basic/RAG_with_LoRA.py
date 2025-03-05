import os
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch


#1️⃣ Menambahkan LoRA untuk reranking → Lebih hemat memori dibanding fine-tuning penuh.
#2️⃣ Menggunakan Cross-Encoder untuk reranking → Lebih akurat dibanding vector similarity biasa.
#3️⃣ Mengirim hasil reranking ke LLM (DeepSeek-R1) → Jawaban lebih akurat karena hanya dokumen terbaik dikirim.


# Define the user's question
query = "Who is Juliet?"

# Define the persistent directory
src = "romeo_and_juliet"
dbName = "chroma_db_huggingface_" + src
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, dbName)

# Define the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Function to query a vector store with similarity search
def query_vector_store(
    store_name, query, embedding_function, search_type, search_kwargs
):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Retrieved Documents ({store_name}) ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
    else:
        print(f"Vector store {store_name} does not exist.")
        return []

    return relevant_docs

# Perform initial retrieval using similarity search
print("\n--- Using Similarity Search ---")
retrieved_docs = query_vector_store(dbName, query, embeddings, "similarity", {"k": 5})

# Load LoRA-enhanced ranking model
print("\n--- Applying LoRA-based Reranking ---")
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
base_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)
ranker_model = get_peft_model(base_model, lora_config)

# Reranking function
def rerank_documents(query, docs):
    inputs = [f"{query} [SEP] {doc.page_content}" for doc in docs]
    tokens = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = ranker_model(**tokens).logits.squeeze().tolist()

    # Sort documents based on score
    reranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return reranked_docs

# Apply reranking
reranked_docs = rerank_documents(query, retrieved_docs)

# Take only top 3 documents after reranking
top_docs = reranked_docs[:3]

# Combine query and top reranked documents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in top_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Load DeepSeek-R1 model with Ollama
model = OllamaLLM(model="deepseek-r1")

# Define messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke model with reranked input
result = model.invoke(messages)

# Display results
print("\n====================================================================================================\n")
print(f'\nQUERY : {query}\n')
print("\n----------------------------------------------------------------------------------------------------\n")
print("--- Generated Response ---")
print("\n----------------------------------------------------------------------------------------------------\n")
print("Result:")
print(result)
print("\n====================================================================================================\n")
