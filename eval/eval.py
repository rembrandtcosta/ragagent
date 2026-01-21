import os
import json
import sys

from tqdm import tqdm
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import (
    faithfulness, context_precision, context_recall, answer_relevancy,
    IDBasedContextRecall, IDBasedContextPrecision
)
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from rag_workflow import process_question

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("eval/qa.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

dataset = []
for item in tqdm(qa, desc="Preparing evaluation dataset"):
    query = item["user_input"]
    reference = item["response"]

    result, footprint = process_question(query)
    relevant_docs = list(
        map(lambda doc: doc.page_content, result["documents"])
    )
    retrieved_docs = list(
        map(lambda doc: doc.metadata.get("source", ""), result["documents"])
    )
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": result["solution"],
            "reference": reference,
            "reference_context_ids": item.get("retrieved_context_ids", []),
            "retrieved_context_ids": retrieved_docs,
        }
    )


evaluation_dataset = EvaluationDataset.from_list(dataset)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.1,
    max_retries=3,
    request_timeout=60
)

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document",
    google_api_key=api_key
)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
        IDBasedContextRecall(),
        IDBasedContextPrecision(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

print(result)
