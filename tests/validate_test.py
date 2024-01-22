import json
import os
from tonic_validate import ValidateScorer, Benchmark, LLMResponse, ValidateApi
from tonic_validate.metrics import AnswerSimilarityMetric, RetrievalPrecisionMetric, AugmentationPrecisionMetric, AugmentationAccuracyMetric, AnswerConsistencyMetric
import requests

from dotenv import load_dotenv

load_dotenv()

def get_llm_response(prompt):
    url = "http://localhost:8000/api/chat"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = { 'Content-Type': 'application/json' }
    response = requests.request("POST", url, headers=headers, data=payload).json()
    result = response['result']
    return result['content'], result['context']

def test_llama_index():
    # Load qa_pairs.json
    qa_pairs = json.load(open('./tests/qa_pairs.json'))
    benchmark = Benchmark(
        questions=[x['question'] for x in qa_pairs],
        answers=[x['answer'] for x in qa_pairs]
    )

    # Save the responses into an array for scoring
    responses = []
    for item in benchmark:
        llm_answer, llm_context_list = get_llm_response(item.question)
        llm_response = LLMResponse(
            llm_answer=llm_answer,
            llm_context_list=llm_context_list,
            benchmark_item=item
        )
        responses.append(llm_response)
    
    # Score run
    metrics = [
        AnswerSimilarityMetric(),
        RetrievalPrecisionMetric(),
        AugmentationPrecisionMetric(),
        AnswerConsistencyMetric(),
        AugmentationAccuracyMetric()
    ]
    scorer = ValidateScorer(metrics)
    run = scorer.score_run(responses)

    # Upload results to web ui
    validate_api = ValidateApi()
    # Get project id from env
    project_id = os.getenv("PROJECT_ID")
    validate_api.upload_run(project_id, run)

    # Check none of the metrics scored too low    
    for metric in metrics:
        if metric.name == AnswerSimilarityMetric.name:
            assert run.overall_scores[metric.name] >= 0
        else:
            assert run.overall_scores[metric.name] >= 0
    
    