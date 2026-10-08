"""Retrieval-only benchmark runner and page-level metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_PATH = Path("benchmarks/rag_agent_benchmark.json")
DEFAULT_CORPUS_DIR = Path("benchmarks")
DEFAULT_OUTPUT_ROOT = Path("benchmark_results")


@dataclass(frozen=True)
class CorpusFile:
    title: str
    short_name: str
    filename: str
    expected_sha256: str
    actual_sha256: str
    path: Path

    @property
    def hash_matches(self) -> bool:
        return self.expected_sha256 == self.actual_sha256

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "short_name": self.short_name,
            "filename": self.filename,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "hash_matches": self.hash_matches,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class RetrievalBenchmarkConfig:
    benchmark_path: Path
    corpus_dir: Path
    output_dir: Path
    k_values: tuple[int, ...]
    allow_hash_mismatch: bool = False


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_retrieval_benchmark(config)


def parse_args(argv: list[str] | None = None) -> RetrievalBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Run the retrieval-only benchmark without calling an LLM."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to rag_agent_benchmark.json",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing exactly the manifest PDFs to index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for result artifacts; defaults to benchmark_results/retrieval_<timestamp>",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="K values to evaluate; defaults to benchmark retrieval_scoring.k_values",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Continue when local PDF hashes differ from the benchmark manifest",
    )
    args = parser.parse_args(argv)

    benchmark = load_json(args.benchmark)
    k_values = tuple(args.k or benchmark["retrieval_scoring"]["k_values"])
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must be positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("K values must be unique")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / f"retrieval_{timestamp}"

    return RetrievalBenchmarkConfig(
        benchmark_path=args.benchmark,
        corpus_dir=args.corpus_dir,
        output_dir=output_dir,
        k_values=tuple(sorted(k_values)),
        allow_hash_mismatch=args.allow_hash_mismatch,
    )


def run_retrieval_benchmark(config: RetrievalBenchmarkConfig) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    benchmark = load_json(config.benchmark_path)
    corpus_files = load_and_verify_corpus_files(
        benchmark=benchmark,
        corpus_dir=config.corpus_dir,
        allow_hash_mismatch=config.allow_hash_mismatch,
    )
    eligible_questions = select_retrieval_questions(benchmark)
    max_k = max(config.k_values)

    project_settings = load_project_settings()
    reset_torch_peak_memory()
    resource_before_index = collect_resource_snapshot()

    pipeline = build_pipeline(project_settings)
    indexing_started = time.perf_counter()
    indexed_documents = []
    for corpus_file in corpus_files:
        chunks = pipeline.ingest(corpus_file.path)
        indexed_documents.append(
            {
                "filename": corpus_file.filename,
                "path": str(corpus_file.path),
                "chunk_count": len(chunks),
                "pages_indexed": sorted(
                    {chunk.page for chunk in chunks if chunk.page is not None}
                ),
            }
        )
    indexing_seconds = time.perf_counter() - indexing_started
    resource_after_index = collect_resource_snapshot()

    per_question_results = []
    retrieval_started = time.perf_counter()
    for question in eligible_questions:
        query_started = time.perf_counter()
        results = pipeline.query(question["question"], max_k)
        query_seconds = time.perf_counter() - query_started
        retrieved_chunks = serialize_results(results, question["required_evidence_groups"])
        per_question_results.append(
            {
                "id": question["id"],
                "category": question.get("category", "unknown"),
                "question": question["question"],
                "required_evidence_groups": question["required_evidence_groups"],
                "query_latency_seconds": query_seconds,
                "metrics_by_k": {
                    str(k): score_question_at_k(
                        question["required_evidence_groups"], retrieved_chunks, k
                    )
                    for k in config.k_values
                },
                "retrieved_chunks": retrieved_chunks,
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_started
    resource_after_retrieval = collect_resource_snapshot()

    aggregate_scores = aggregate_scores_by_group(per_question_results, config.k_values)
    latency = summarize_latency(
        [result["query_latency_seconds"] for result in per_question_results],
        indexing_seconds=indexing_seconds,
        retrieval_seconds=retrieval_seconds,
    )
    metadata = build_run_metadata(
        benchmark=benchmark,
        benchmark_path=config.benchmark_path,
        corpus_files=corpus_files,
        k_values=config.k_values,
        indexed_documents=indexed_documents,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        latency=latency,
        project_settings=project_settings,
        resource_measurements={
            "before_index": resource_before_index,
            "after_index": resource_after_index,
            "after_retrieval": resource_after_retrieval,
        },
        allow_hash_mismatch=config.allow_hash_mismatch,
    )
    payload = {
        "metadata": metadata,
        "aggregate_scores": aggregate_scores,
        "per_question_results": per_question_results,
    }
    write_artifacts(config.output_dir, payload)
    print(f"Wrote retrieval benchmark artifacts to {config.output_dir}")
    print(json.dumps(aggregate_scores["all_retrieval_eligible"], indent=2))
    return payload


def load_project_settings() -> Any:
    from app.core.config import settings

    return settings


def build_pipeline(project_settings: Any) -> Any:
    from app.rag.chunker import TextChunker
    from app.rag.document_loader import DocumentLoader
    from app.rag.embeddings import EmbeddingService
    from app.rag.pipeline import RetrievalPipeline
    from app.rag.vector_store import FaissVectorStore
    from app.rag.bm25_store import BM25Store
    from app.rag.retriever import BM25Retriever, VectorRetriever

    embedding_service = EmbeddingService(project_settings.embedding_model_name)
    vector_store = FaissVectorStore()
    bm25_store = BM25Store(project_settings.bm25_k1, project_settings.bm25_b)
    retriever = (
        BM25Retriever(bm25_store)
        if project_settings.retrieval_mode == "bm25"
        else VectorRetriever(embedding_service, vector_store)
    )
    return RetrievalPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(
            project_settings.chunk_size, project_settings.chunk_overlap
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
        retriever=retriever,
        bm25_store=bm25_store,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_verify_corpus_files(
    *, benchmark: dict[str, Any], corpus_dir: Path, allow_hash_mismatch: bool
) -> list[CorpusFile]:
    manifest = benchmark.get("corpus_manifest", [])
    if len(manifest) != 3:
        raise ValueError(f"Expected exactly 3 corpus files, found {len(manifest)}")

    corpus_files = []
    errors = []
    for entry in manifest:
        path = corpus_dir / entry["filename"]
        if not path.is_file():
            errors.append(f"Missing corpus file: {path}")
            continue
        actual_sha256 = sha256_file(path)
        corpus_file = CorpusFile(
            title=entry.get("title", ""),
            short_name=entry.get("short_name", ""),
            filename=entry["filename"],
            expected_sha256=entry["sha256"],
            actual_sha256=actual_sha256,
            path=path,
        )
        if not corpus_file.hash_matches:
            errors.append(
                f"Hash mismatch for {path}: expected {corpus_file.expected_sha256}, "
                f"got {corpus_file.actual_sha256}"
            )
        corpus_files.append(corpus_file)

    if errors and not allow_hash_mismatch:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            "Corpus validation failed. Replace the PDFs or rerun with "
            f"--allow-hash-mismatch for debugging only.\n{joined_errors}"
        )
    if len(corpus_files) != len(manifest):
        raise ValueError("Cannot continue because one or more corpus files are missing")
    return corpus_files


def select_retrieval_questions(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    eligible_ids = set(benchmark["retrieval_scoring"]["eligible_question_ids"])
    questions_by_id = {question["id"]: question for question in benchmark["questions"]}
    missing_ids = sorted(eligible_ids - set(questions_by_id))
    if missing_ids:
        raise ValueError(f"Eligible question IDs are missing: {missing_ids}")

    selected = []
    for question_id in benchmark["retrieval_scoring"]["eligible_question_ids"]:
        question = questions_by_id[question_id]
        groups = question.get("required_evidence_groups")
        if not groups:
            raise ValueError(f"Question {question_id} has no required evidence groups")
        selected.append(question)
    return selected


def serialize_results(
    results: list[Any], evidence_groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "filename": result.chunk.filename,
            "page": result.chunk.page,
            "chunk_id": result.chunk.chunk_id,
            "score": result.score,
            "matched_group_ids": sorted(
                matching_group_ids(result.chunk.filename, result.chunk.page, evidence_groups)
            ),
            "text": result.chunk.text,
            "metadata": result.chunk.metadata,
        }
        for rank, result in enumerate(results, start=1)
    ]


def score_question_at_k(
    evidence_groups: list[dict[str, Any]], retrieved_chunks: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    total_groups = len(evidence_groups)
    if total_groups == 0:
        return {
            "page_hit": 0.0,
            "page_recall": 0.0,
            "page_mrr": 0.0,
            "page_ndcg": 0.0,
            "complete_evidence": 0.0,
            "covered_group_ids": [],
            "first_match_rank": None,
        }

    covered_group_ids: set[str] = set()
    first_match_rank: int | None = None
    dcg = 0.0
    for chunk in retrieved_chunks[:k]:
        rank = int(chunk["rank"])
        new_matches = set(chunk["matched_group_ids"]) - covered_group_ids
        if new_matches:
            if first_match_rank is None:
                first_match_rank = rank
            dcg += len(new_matches) / math.log2(rank + 1)
            covered_group_ids.update(new_matches)

    idcg = ideal_dcg(evidence_groups, k)
    recall = len(covered_group_ids) / total_groups
    return {
        "page_hit": 1.0 if covered_group_ids else 0.0,
        "page_recall": recall,
        "page_mrr": 1.0 / first_match_rank if first_match_rank else 0.0,
        "page_ndcg": dcg / idcg if idcg else 0.0,
        "complete_evidence": 1.0 if len(covered_group_ids) == total_groups else 0.0,
        "covered_group_ids": sorted(covered_group_ids),
        "first_match_rank": first_match_rank,
    }


def aggregate_scores_by_group(
    per_question_results: list[dict[str, Any]], k_values: tuple[int, ...]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all_retrieval_eligible": per_question_results,
        "narrative_retrieval": [
            result
            for result in per_question_results
            if result["category"] == "narrative_retrieval" or result["id"] == "Q27"
        ],
        "table_quantitative": [
            result
            for result in per_question_results
            if result["category"] == "table_quantitative"
        ],
    }
    for result in per_question_results:
        groups.setdefault(f"category:{result['category']}", []).append(result)

    return {
        group_name: aggregate_one_group(results, k_values)
        for group_name, results in groups.items()
        if results
    }


def aggregate_one_group(
    results: list[dict[str, Any]], k_values: tuple[int, ...]
) -> dict[str, Any]:
    metric_names = [
        "page_hit",
        "page_recall",
        "page_mrr",
        "page_ndcg",
        "complete_evidence",
    ]
    return {
        "question_count": len(results),
        "question_ids": [result["id"] for result in results],
        "metrics_by_k": {
            str(k): {
                metric_name: mean(
                    result["metrics_by_k"][str(k)][metric_name]
                    for result in results
                )
                for metric_name in metric_names
            }
            for k in k_values
        },
    }


def matching_group_ids(
    filename: str, page: int | None, evidence_groups: list[dict[str, Any]]
) -> set[str]:
    if page is None:
        return set()
    group_ids = set()
    for group in evidence_groups:
        for candidate in group.get("any_of", []):
            if candidate.get("filename") == filename and candidate.get("pdf_page") == page:
                group_ids.add(group["group_id"])
                break
    return group_ids


def ideal_dcg(evidence_groups: list[dict[str, Any]], k: int) -> float:
    group_ids = tuple(group["group_id"] for group in evidence_groups)
    page_to_groups: dict[tuple[str, int], set[str]] = {}
    for group in evidence_groups:
        for candidate in group.get("any_of", []):
            key = (candidate["filename"], int(candidate["pdf_page"]))
            page_to_groups.setdefault(key, set()).add(group["group_id"])
    candidate_covers = tuple(frozenset(groups) for groups in page_to_groups.values())

    @lru_cache(maxsize=None)
    def best_dcg(remaining: frozenset[str], rank: int) -> float:
        if not remaining or rank > k:
            return 0.0
        discount = math.log2(rank + 1)
        best = 0.0
        for cover in candidate_covers:
            gained = remaining.intersection(cover)
            if not gained:
                continue
            score = len(gained) / discount + best_dcg(
                frozenset(remaining.difference(gained)), rank + 1
            )
            best = max(best, score)
        return best

    return best_dcg(frozenset(group_ids), 1)


def build_run_metadata(
    *,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    corpus_files: list[CorpusFile],
    k_values: tuple[int, ...],
    indexed_documents: list[dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
    latency: dict[str, Any],
    project_settings: Any,
    resource_measurements: dict[str, Any],
    allow_hash_mismatch: bool,
) -> dict[str, Any]:
    corpus_hashes = [corpus_file.to_json() for corpus_file in corpus_files]
    return {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("version"),
        "benchmark_sha256": sha256_file(benchmark_path),
        "run_started_at": started_at.isoformat(),
        "run_finished_at": finished_at.isoformat(),
        "corpus_hash_algorithm": "sha256",
        "corpus_sha256": combined_corpus_sha256(corpus_files),
        "corpus_files": corpus_hashes,
        "corpus_hashes_verified": all(item["hash_matches"] for item in corpus_hashes),
        "allow_hash_mismatch": allow_hash_mismatch,
        "embedding_model_name": project_settings.embedding_model_name,
        "embedding_model_revision": "unknown",
        "chunk_size": project_settings.chunk_size,
        "chunk_overlap": project_settings.chunk_overlap,
        "k_values": list(k_values),
        "retrieval_configuration": {
            "mode": project_settings.retrieval_mode,
            "bm25_k1": project_settings.bm25_k1,
            "bm25_b": project_settings.bm25_b,
            "bm25_tokenizer": "casefold + Unicode word tokens; no stemming",
            "pipeline": "RetrievalPipeline",
            "loader": "DocumentLoader",
            "chunker": "TextChunker",
            "embedding_service": "EmbeddingService",
            "vector_store": "FaissVectorStore",
            "similarity": (
                "BM25Okapi relevance"
                if project_settings.retrieval_mode == "bm25"
                else "cosine via normalized inner product"
            ),
            "query_strategy": "run max K once per question and score prefixes",
            "score_unit": "filename_and_one_based_pdf_page",
            "page_ndcg_definition": (
                "A chunk gains one point for each previously uncovered required "
                "evidence group matched by its filename/page. Ideal DCG is the "
                "optimal page sequence over labeled evidence alternatives."
            ),
        },
        "indexed_documents": indexed_documents,
        "latency_seconds": latency,
        "resource_measurements": resource_measurements,
        "dependency_versions": dependency_versions(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }


def summarize_latency(
    query_latencies: list[float], *, indexing_seconds: float, retrieval_seconds: float
) -> dict[str, Any]:
    return {
        "indexing_total": indexing_seconds,
        "retrieval_total": retrieval_seconds,
        "query_count": len(query_latencies),
        "query_mean": mean(query_latencies),
        "query_median": statistics.median(query_latencies) if query_latencies else 0.0,
        "query_min": min(query_latencies) if query_latencies else 0.0,
        "query_max": max(query_latencies) if query_latencies else 0.0,
    }


def write_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_metadata.json", payload["metadata"])
    write_json(output_dir / "aggregate_scores.json", payload["aggregate_scores"])
    write_json(output_dir / "per_question_results.json", payload["per_question_results"])
    with (output_dir / "retrieved_chunks.jsonl").open("w", encoding="utf-8") as file:
        for question_result in payload["per_question_results"]:
            for chunk in question_result["retrieved_chunks"]:
                file.write(
                    json.dumps(
                        {
                            "question_id": question_result["id"],
                            "category": question_result["category"],
                            "question": question_result["question"],
                            **chunk,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_corpus_sha256(corpus_files: list[CorpusFile]) -> str:
    digest = hashlib.sha256()
    for corpus_file in sorted(corpus_files, key=lambda item: item.filename):
        digest.update(f"{corpus_file.filename}:{corpus_file.actual_sha256}\n".encode())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    distributions = [
        "pydantic",
        "python-dotenv",
        "PyMuPDF",
        "sentence-transformers",
        "faiss-cpu",
        "rank-bm25",
        "numpy",
        "ollama",
        "torch",
    ]
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def collect_resource_snapshot() -> dict[str, Any]:
    return {
        "nvidia_smi": nvidia_smi_memory(),
        "torch_cuda": torch_cuda_memory(),
    }


def nvidia_smi_memory() -> list[dict[str, Any]] | str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
            }
        )
    return rows


def torch_cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch cuda unavailable"}
    device_index = torch.cuda.current_device()
    return {
        "available": True,
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "memory_allocated_mb": torch.cuda.memory_allocated(device_index) / 1024**2,
        "memory_reserved_mb": torch.cuda.memory_reserved(device_index) / 1024**2,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device_index)
        / 1024**2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device_index)
        / 1024**2,
    }


def reset_torch_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)
