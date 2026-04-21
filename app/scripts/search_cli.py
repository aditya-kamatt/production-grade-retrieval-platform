import argparse
import json
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LEC Search API")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/search")
    args = parser.parse_args()

    payload = {
        "query": args.query,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "use_reranker": not args.no_reranker,
    }

    response = requests.post(args.url, json=payload, timeout=30)
    response.raise_for_status()

    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()