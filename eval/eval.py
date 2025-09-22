import time
import os
import json
import sys
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_workflow import process_question

with open("eval/qa.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

print(qa)

dataset = []

for item in qa:
    query = item["user_input"]
    reference = item["response"]

    result = process_question(query)
    print(result)
    # time.sleep(25)
    relevant_docs = list(map(lambda doc: doc.page_content, result["documents"]))
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": result["solution"],
            "reference": reference
        }
    )


evaluation_dataset = EvaluationDataset.from_list(dataset)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

llm = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key,
    temperature=0.1,
    max_retries=3,
    request_timeout=60
)

# llm = ChatOllama(model="llama3.2:1b", temperature=0)

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document",
    google_api_key=api_key
)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# os.makedirs("eval/results", exist_ok=True)

# with open("eval/results/ragas_evaluation_results.json", "w", encoding="utf-8") as f:
#    json.dump(result.to_json(), f, indent=2, ensure_ascii=False)
print(result)

