import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from app.rag_system import RAGSystem
from datasets import load_dataset


# Ignore the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

#load the hugging face document for evaluation
knowledge_base = load_dataset("m-ric/huggingface_doc", split="train")

print("type of dataset:", type(knowledge_base))
print("length of dataset:", len(knowledge_base))


# Initialize RAG system
rag = RAGSystem(use_mlflow=False)

id = 1

for row in knowledge_base:
    print("Ingesting document")
    rag.ingest_document(
        source=row['text'],
        source_type="text",
        metadata={"id": id, "source": row["source"].split("/")[1], "category": "HF"},
    )
    id = id + 1

'''

# Ingest a document
result = rag.ingest_document(
    source="https://huggingface.co/datasets/m-ric/huggingface_doc/embed/viewer/default/train",
    source_type="df",
    metadata={"author": "Daily Dish", "category": "FAQ"},
)
'''

stats = rag.get_collection_stats()
print(f"Collection stats: {stats}")

# Query the system
#response = rag.answer_query("What is an LLM?")

'''
#Evaluation against ground truth
ragas_dataset = rag.create_ground_truth("datasets/hf_doc_qa_eval.csv")

print(f"Created RAGAS dataset with {len(ragas_dataset)} examples")
print(ragas_dataset["question"][0])
print(ragas_dataset["expected_answer"][0])


print(f"Answer: {response.answer}")
print(f"Retrieved {len(response.retrieved_documents)} documents")
print(f"Reranked {len(response.reranked_documents)} documents")
'''