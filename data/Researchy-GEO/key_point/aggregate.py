import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, List
from tqdm.asyncio import tqdm_asyncio
from pydantic import create_model
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class for a single point
class KeyPointAggregationSingle(BaseModel):
    point_number: int
    point_content: str
    original_point_number: List[int]

# class for all key point extraction results
class KeyPointAggregation(BaseModel):
    points: List[KeyPointAggregationSingle]

def create_prompt(original_points: List[str]):

    original_points_with_number = {i + 1: point for i, point in enumerate(original_points)}

    return f"""
You are given a list of key points extracted from multiple documents. Your task is to aggregate these points according to the following instructions:

1. Identify and deduplicate any duplicated or redundant points. Merge them into a single, representative point.
2. Identify contradictory points. Merge them into a single point that presents both sides, e.g., "Sources claim that X, while other sources claim that Y".

IMPORTANT RULES:
- Every aggregated point must preserve **all original information** from the included points.
- Do not invent or add any new information. Only use what is already present.
- Do not provide any explanations or summaries beyond the aggregation itself.
- Each aggregated point should **capture a single atomic idea**. Avoid combining unrelated aspects into one point.
- Keep the aggregated point **concise but complete**: include all essential details needed to fully represent the merged idea, but do not make it overly detailed or verbose.
- For each aggregated point, include a reference to the original point numbers it is based on, e.g., "original_point_number": [1, 3, 7].

Respond strictly in JSON format:
{{
    "points": [
        {{
            "point_number": point_number,
            "point_content": point_content,
            "original_point_number": [original_point_number1, original_point_number2, ...]
        }},
        ...
    ]
}}

[Original Points]
{original_points_with_number}
"""
    
async def aggregate_single_query(semaphore, key_point_dir, qid, model):

    key_point_dir = Path(key_point_dir)
    key_point_path = key_point_dir / f"{qid}.json"
    aggregated_key_point_path = key_point_dir / f"{qid}_aggregated.json"


    with open(key_point_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        key_points = data["key_points"]
        question = data["question"]
        if key_points is None:
            print(f"Key points are None for {key_point_path}")


    all_points = []

    for cluewebID, points in key_points.items():
        for point in points:
            all_points.append(point["point_content"])

    # aggregate key points
    prompt = create_prompt(all_points)
    chat_pattern = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    async with semaphore:
        response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=chat_pattern,
                    response_format=KeyPointAggregation,
                    temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        aggregated_points = result['points']

        results = {
            "question": question,
            "key_points": aggregated_points
        }

        with open(aggregated_key_point_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    


async def aggregate_all_queries(key_point_dir, model):

    semaphore = asyncio.Semaphore(100)

    query_ids = []
    for p in Path(key_point_dir).glob("*.json"):
        if p.name.endswith("_aggregated.json"):
            continue  
        aggregated_file = p.with_name(f"{p.stem}_aggregated.json")
        if not aggregated_file.exists():
            query_ids.append(p.stem)

    tasks = [aggregate_single_query(semaphore, key_point_dir, qid, model) for qid in query_ids]
    
    await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    
    key_point_dir = "key_point"
    model = "gpt-4.1-mini"

    asyncio.run(aggregate_all_queries(key_point_dir, model))
