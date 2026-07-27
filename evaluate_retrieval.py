"""运行小型、可重复的检索回归集。

该脚本只调用嵌入接口，不调用聊天模型。上线前应在业务问题集上扩展此文件，
并把阈值调整建立在测量结果上，而不是凭感觉修改。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config_data as config
from model_clients import create_embeddings
from vector_stores import VectorStore


def _source(document: Any) -> str | None:
    value = (document.metadata or {}).get("source")
    return str(value) if value else None


def evaluate(dataset_path: Path) -> dict[str, Any]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评估集必须是非空 JSON 数组")

    vector_store = VectorStore(create_embeddings())
    rows = []
    passed = 0

    for case in cases:
        query = str(case["query"])
        expected_source = case.get("expected_source")
        raw_results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=config.retrieval_k,
            score_threshold=0.0,
        )
        qualified = [
            (document, float(score))
            for document, score in raw_results
            if float(score) >= config.similarity_score_threshold
        ]
        sources = [_source(document) for document, _ in qualified]
        if expected_source is None:
            case_passed = not qualified
        else:
            case_passed = expected_source in sources
        passed += int(case_passed)
        rows.append(
            {
                "id": case.get("id"),
                "passed": case_passed,
                "expected_source": expected_source,
                "qualified_sources": sources,
                "top_results": [
                    {
                        "source": _source(document),
                        "chunk_index": (document.metadata or {}).get("chunk_index"),
                        "score": round(float(score), 4),
                    }
                    for document, score in raw_results
                ],
            }
        )

    return {
        "collection": config.collection_name,
        "k": config.retrieval_k,
        "score_threshold": config.similarity_score_threshold,
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(passed / len(rows), 4),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 RAG 检索回归")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().with_name("eval_dataset.json"),
    )
    args = parser.parse_args()
    try:
        report = evaluate(args.dataset.resolve())
    except (RuntimeError, ValueError) as exc:
        print(f"评估无法运行：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
