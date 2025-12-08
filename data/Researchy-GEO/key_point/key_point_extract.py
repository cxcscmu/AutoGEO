import asyncio
import json
from pathlib import Path
from typing import Literal, List
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class for a single point
class KeyPointExtractionSingle(BaseModel):
    point_number: int
    point_content: str
    spans: List[str]

# class for all key point extraction results
class KeyPointExtraction(BaseModel):
    points: List[KeyPointExtractionSingle]

def create_prompt(question, text):

    return f"""Based on the text provided, identify key points in the text that directly help in responding to the query. The key points are not simply some key content of the text, but rather the key points that are important for **answering the query**.

IMPORTANT: Ensure each point is helpful in responding to the query. Keep the point using the original language and do not add explanations.
IMPORTANT: Each span must be a single consecutive verbatim span from the corresponding passages. Copy verbatim the spans, don't modify any word!

Respond strictly in JSON format:
{{
    "points": [
        {{
            "point_number": point_number,
            "point_content": point_content,
            "spans": [span1, span2, ...]
        }},
        ...
    ]
}}

Remember:
- key points can be abstracted or summarized, but the span must be a copy of the original text. The content of the key point does NOT need to be the same as that of the span.
- These key points must be helpful in responding to the query.
- If there are multiple spans for a point, add all of them in the spans list.

[Query]: {question}
[Text]: {text}

"""


async def extract_key_point(semaphore, question, CluewebID, text, model):
    async with semaphore:
        prompt = create_prompt(question, text)
        chat_pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await client.beta.chat.completions.parse(
                        model=model,
                        messages=chat_pattern,
                        response_format=KeyPointExtraction,
                        temperature=0
            )
            result = json.loads(response.choices[0].message.content)
            return CluewebID, result['points']
        except Exception as e:
            print(f"Error processing {CluewebID}: {e}")
            return CluewebID, None
    
async def extract_key_point_single_query(semaphore, result_path, qid, question, CluewebIDs, texts, model):
    tasks = [extract_key_point(semaphore, question, CluewebID, text, model) for CluewebID, text in zip(CluewebIDs, texts)]
    key_points = await tqdm_asyncio.gather(*tasks)
    key_points = dict(key_points)
    
    # remove None CluewebIDs
    key_points = {k: v for k, v in key_points.items() if v is not None}

    extract_result_path = Path(result_path) / f"{qid}.json"

    results = {
            "question": question,
            "key_points": key_points
        }

    with open(extract_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

async def extract_all_queries(sampled_queries, result_path, model):

    semaphore = asyncio.Semaphore(100)
    tasks = []

    for data in sampled_queries:
        question = data["query"]
        id = data["id"]
        doc_streams = data["DocStream"]
        CluewebIDs = []
        texts = []
        for doc_stream in doc_streams:
            CluewebIDs.append(doc_stream["CluewebID"])
            texts.append(doc_stream["CluewebDocument"])
        
        tasks.append(extract_key_point_single_query(semaphore, result_path, id, question, CluewebIDs, texts, model))

    await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    
    with open("queries/researchy_queries_sample_clueweb.jsonl", "r") as f:
        sampled_queries = [json.loads(line) for line in f]
    
    key_point_dir = "key_point"
    model = "gpt-4.1-mini"

    asyncio.run(extract_all_queries(sampled_queries, key_point_dir, model))
