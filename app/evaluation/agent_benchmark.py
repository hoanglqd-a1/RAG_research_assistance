"""Agent-loop benchmark runner for behavior, tools, citations, and latency."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.retrieval_benchmark import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_CORPUS_DIR,
    DEFAULT_OUTPUT_ROOT,
    CorpusFile,
    collect_resource_snapshot,
    combined_corpus_sha256,
    dependency_versions,
    load_and_verify_corpus_files,
    load_json,
    matching_group_ids,
    mean,
    reset_torch_peak_memory,
    sha256_file,
    write_json,
)

AVAILABLE_AGENT_TOOLS = (
    "search_documents",
    "calculator",
    "get_document_metadata",
    "summarize_document",
)
DEFAULT_RECOMMENDED_GROUPS = (
    "summary",
    "narrative_retrieval",
    "table_quantitative",
    "abstention",
    "agent_behavior",
)
DEFERRED_WEB_GROUP = "deferred_web"

CITATION_RE = re.compile(
    r"\[?\b(?P<filename>[A-Za-z0-9_.-]+\.pdf)\b\s*,?\s*"
    r"(?:page|p\.?|pp\.?)\s*(?P<page>\d+)\]?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

ABSTENTION_TERMS = (
    "not report",
    "not reported",
    "does not report",
    "doesn't report",
    "not mention",
    "not specified",
    "not provided",
    "not available",
    "no mobile",
    "cannot determine",
    "can't determine",
    "insufficient",
)
EXTERNAL_LIMIT_TERMS = (
    "web",
    "internet",
    "external",
    "current",
    "today",
    "up-to-date",
    "up to date",
    "cannot verify",
    "can't verify",
    "outside",
    "indexed corpus",
    "indexed documents",
)
FALSE_PREMISE_TERMS = (
    "premise",
    "incorrect",
    "not correct",
    "does not rely",
    "doesn't rely",
    "does not use colmap",
    "doesn't use colmap",
    "without colmap",
    "avoid colmap",
    "obviates",
)
NOT_FOUND_TERMS = (
    "not indexed",
    "no indexed document",
    "not found",
    "could not find",
    "can't find",
    "does not exist",
    "is not available",
    "isn't indexed",
)


@dataclass(frozen=True)
class AgentBenchmarkConfig:
    benchmark_path: Path
    corpus_dir: Path
    output_dir: Path
    include_deferred_web: bool = False
    case_ids: tuple[str, ...] | None = None
    allow_hash_mismatch: bool = False
    judge_faithfulness: bool = False
    supporting_context_chars: int = 6000


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_agent_benchmark(config)


def parse_args(argv: list[str] | None = None) -> AgentBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generation-backed agent benchmark for tool selection, "
            "behavior, citations, latency, and resource measurements."
        )
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
        help="Directory for result artifacts; defaults to benchmark_results/agent_<timestamp>",
    )
    parser.add_argument(
        "--case-id",
        nargs="+",
        default=None,
        help="Optional explicit case IDs to run, e.g. Q26 B01 B04",
    )
    parser.add_argument(
        "--include-deferred-web",
        action="store_true",
        help="Also run Q28-Q30 as capability-limit checks; they are excluded by default.",
    )
    parser.add_argument(
        "--judge-faithfulness",
        action="store_true",
        help=(
            "Use the configured local Ollama model as an optional faithfulness "
            "judge over saved supporting chunks."
        ),
    )
    parser.add_argument(
        "--supporting-context-chars",
        type=int,
        default=6000,
        help="Maximum source text characters saved per case for faithfulness review.",
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Continue when local PDF hashes differ from the benchmark manifest",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / f"agent_{timestamp}"

    if args.supporting_context_chars <= 0:
        raise ValueError("--supporting-context-chars must be greater than zero")

    case_ids = tuple(args.case_id) if args.case_id else None
    return AgentBenchmarkConfig(
        benchmark_path=args.benchmark,
        corpus_dir=args.corpus_dir,
        output_dir=output_dir,
        include_deferred_web=args.include_deferred_web,
        case_ids=case_ids,
        allow_hash_mismatch=args.allow_hash_mismatch,
        judge_faithfulness=args.judge_faithfulness,
        supporting_context_chars=args.supporting_context_chars,
    )


def run_agent_benchmark(config: AgentBenchmarkConfig) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    benchmark = load_json(config.benchmark_path)
    corpus_files = load_and_verify_corpus_files(
        benchmark=benchmark,
        corpus_dir=config.corpus_dir,
        allow_hash_mismatch=config.allow_hash_mismatch,
    )
    cases = select_agent_cases(
        benchmark,
        include_deferred_web=config.include_deferred_web,
        case_ids=config.case_ids,
    )

    project_settings = load_project_settings()
    reset_torch_peak_memory()
    resource_before_index = collect_resource_snapshot()

    services = build_application_services(project_settings)
    indexing_started = time.perf_counter()
    indexed_documents = []
    for corpus_file in corpus_files:
        chunks = services.pipeline.ingest(corpus_file.path)
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
    chunk_catalog = build_chunk_catalog(services.vector_store)
    resource_after_index = collect_resource_snapshot()
    faithfulness_judge = (
        build_faithfulness_judge(project_settings)
        if config.judge_faithfulness
        else None
    )

    per_case_results = []
    agent_started = time.perf_counter()
    for case in cases:
        case_started = time.perf_counter()
        try:
            result = services.agent.run(case["question"])
            case_latency = time.perf_counter() - case_started
            case_result = serialize_successful_case(
                case,
                result,
                case_latency,
                chunk_catalog,
                config.supporting_context_chars,
            )
            if faithfulness_judge is not None:
                judge_started = time.perf_counter()
                try:
                    case_result["faithfulness_scoring"] = score_faithfulness_with_judge(
                        case_result, faithfulness_judge
                    )
                except Exception as judge_exc:
                    case_result["faithfulness_scoring"] = {
                        "mode": "ollama_judge_error",
                        "faithful": None,
                        "score": None,
                        "error": f"{type(judge_exc).__name__}: {judge_exc}",
                    }
                case_result["faithfulness_scoring"]["judge_latency_seconds"] = (
                    time.perf_counter() - judge_started
                )
            per_case_results.append(case_result)
        except Exception as exc:  # Keep the benchmark artifact useful on model failure.
            case_latency = time.perf_counter() - case_started
            per_case_results.append(serialize_failed_case(case, exc, case_latency))
    agent_seconds = time.perf_counter() - agent_started
    resource_after_agent = collect_resource_snapshot()

    aggregate_scores = aggregate_agent_scores(per_case_results)
    latency = summarize_agent_latency(
        [result["latency_seconds"] for result in per_case_results],
        indexing_seconds=indexing_seconds,
        agent_seconds=agent_seconds,
    )
    metadata = build_agent_run_metadata(
        benchmark=benchmark,
        benchmark_path=config.benchmark_path,
        corpus_files=corpus_files,
        cases=cases,
        indexed_documents=indexed_documents,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        latency=latency,
        project_settings=project_settings,
        resource_measurements={
            "before_index": resource_before_index,
            "after_index": resource_after_index,
            "after_agent": resource_after_agent,
        },
        allow_hash_mismatch=config.allow_hash_mismatch,
        include_deferred_web=config.include_deferred_web,
        explicit_case_ids=config.case_ids,
        judge_faithfulness=config.judge_faithfulness,
        supporting_context_chars=config.supporting_context_chars,
    )
    payload = {
        "metadata": metadata,
        "aggregate_scores": aggregate_scores,
        "per_case_results": per_case_results,
    }
    write_agent_artifacts(config.output_dir, payload)
    print(f"Wrote agent benchmark artifacts to {config.output_dir}")
    print(json.dumps(aggregate_scores.get("main_non_web", {}), indent=2))
    return payload


def load_project_settings() -> Any:
    from app.core.config import settings

    return settings


def build_application_services(project_settings: Any) -> Any:
    from app.services import build_services

    return build_services(project_settings)


def build_faithfulness_judge(project_settings: Any) -> Any:
    from app.llm.ollama_client import OllamaClient

    return OllamaClient(project_settings.llm_model, host=project_settings.ollama_host)


def select_agent_cases(
    benchmark: dict[str, Any],
    *,
    include_deferred_web: bool,
    case_ids: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    cases_by_id: dict[str, dict[str, Any]] = {}
    for question in benchmark.get("questions", []):
        cases_by_id[question["id"]] = {**question, "case_source": "questions"}
    for behavior in benchmark.get("behavior_tests", []):
        cases_by_id[behavior["id"]] = {**behavior, "case_source": "behavior_tests"}

    recommended_group_by_id = recommended_groups_by_case_id(benchmark)
    if case_ids:
        ordered_ids = list(case_ids)
    else:
        groups = list(DEFAULT_RECOMMENDED_GROUPS)
        if include_deferred_web:
            groups.append(DEFERRED_WEB_GROUP)
        ordered_ids = [
            case_id
            for group in groups
            for case_id in benchmark.get("recommended_first_run", {}).get(group, [])
        ]

    missing_ids = [case_id for case_id in ordered_ids if case_id not in cases_by_id]
    if missing_ids:
        raise ValueError(f"Agent benchmark case IDs are missing: {missing_ids}")

    selected = []
    for case_id in ordered_ids:
        case = dict(cases_by_id[case_id])
        case["reporting_group"] = recommended_group_by_id.get(
            case_id, case.get("category", "unknown")
        )
        case["derived_evidence_groups"] = evidence_groups_for_case(case)
        case["tool_policy"] = tool_policy_for_case(case)
        selected.append(case)
    return selected


def recommended_groups_by_case_id(benchmark: dict[str, Any]) -> dict[str, str]:
    group_by_id = {}
    for group, case_ids in benchmark.get("recommended_first_run", {}).items():
        for case_id in case_ids:
            group_by_id.setdefault(case_id, group)
    return group_by_id


def evidence_groups_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    required_groups = case.get("required_evidence_groups")
    if required_groups:
        return required_groups

    groups = []
    for index, evidence in enumerate(case.get("evidence", []), start=1):
        filename = evidence.get("filename")
        page = evidence.get("pdf_page")
        if filename and page is not None:
            groups.append(
                {
                    "group_id": f"evidence_{index:02d}",
                    "any_of": [{"filename": filename, "pdf_page": int(page)}],
                }
            )
    return groups


def tool_policy_for_case(case: dict[str, Any]) -> dict[str, Any]:
    category = case.get("category")
    subtype = case.get("subtype")
    requires_calculation = bool(case.get("requires_calculation"))
    requires_web_search = bool(case.get("requires_web_search"))
    tool_selection = case.get("tool_selection_scoring") or {}

    if tool_selection.get("required_tools"):
        plans = [list(tool_selection["required_tools"])]
        allowed = sorted(set(plans[0]))
        return {
            "accepted_tool_plans": plans,
            "allowed_tools": allowed,
            "requires_no_tools": False,
            "reason": "explicit_tool_selection_scoring",
        }

    if subtype == "metadata_lookup":
        return policy([["get_document_metadata"]], ["get_document_metadata"], "metadata_lookup")
    if subtype == "invalid_document":
        return policy(
            [["summarize_document"], ["get_document_metadata"]],
            ["summarize_document", "get_document_metadata"],
            "invalid_document",
        )
    if subtype == "no_tool_needed":
        return policy([[]], [], "direct_answer", requires_no_tools=True)
    if subtype == "simple_calculation":
        return policy([["calculator"]], ["calculator"], "simple_calculation")

    if requires_web_search:
        return policy([[]], ["search_documents"], "deferred_web_limit")
    if category == "summary":
        return policy(
            [["summarize_document"], ["search_documents"]],
            ["summarize_document", "search_documents"],
            "summary",
        )
    if category == "table_quantitative" and requires_calculation:
        return policy(
            [["search_documents", "calculator"]],
            ["search_documents", "calculator"],
            "retrieve_and_calculate",
        )
    if category in {"narrative_retrieval", "abstention_and_false_premise"}:
        return policy(
            [["search_documents"]],
            ["search_documents", "calculator"],
            "document_retrieval",
        )

    return policy([[]], list(AVAILABLE_AGENT_TOOLS), "no_specific_policy")


def policy(
    accepted_tool_plans: list[list[str]],
    allowed_tools: list[str],
    reason: str,
    *,
    requires_no_tools: bool = False,
) -> dict[str, Any]:
    return {
        "accepted_tool_plans": accepted_tool_plans,
        "allowed_tools": sorted(set(allowed_tools)),
        "requires_no_tools": requires_no_tools,
        "reason": reason,
    }


def build_chunk_catalog(vector_store: Any) -> dict[str, Any]:
    by_chunk_id: dict[tuple[str, str], dict[str, Any]] = {}
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for chunk in vector_store.list_chunks():
        item = {
            "filename": chunk.filename,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        if chunk.chunk_id:
            by_chunk_id[(chunk.filename, chunk.chunk_id)] = item
        if chunk.page is not None:
            by_page.setdefault((chunk.filename, chunk.page), []).append(item)
    return {"by_chunk_id": by_chunk_id, "by_page": by_page}


def supporting_chunks_for_sources(
    sources: list[dict[str, Any]], chunk_catalog: dict[str, Any], max_chars: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    total_chars = 0

    def add_chunk(chunk: dict[str, Any]) -> None:
        nonlocal total_chars
        if total_chars >= max_chars:
            return
        key = (chunk["filename"], chunk.get("chunk_id"))
        if key in seen:
            return
        seen.add(key)
        remaining = max_chars - total_chars
        text = chunk["text"]
        clipped_text = text[:remaining]
        chunks.append(
            {
                "filename": chunk["filename"],
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "text": clipped_text,
                "truncated": len(clipped_text) < len(text),
                "metadata": chunk.get("metadata", {}),
            }
        )
        total_chars += len(clipped_text)

    by_chunk_id = chunk_catalog["by_chunk_id"]
    by_page = chunk_catalog["by_page"]
    for source in sources:
        chunk_id = source.get("chunk_id")
        if chunk_id:
            chunk = by_chunk_id.get((source["filename"], chunk_id))
            if chunk:
                add_chunk(chunk)
    for source in sources:
        page = source.get("page")
        if page is None:
            continue
        for chunk in by_page.get((source["filename"], page), []):
            add_chunk(chunk)
    return chunks


def serialize_successful_case(
    case: dict[str, Any],
    result: Any,
    latency_seconds: float,
    chunk_catalog: dict[str, Any],
    supporting_context_chars: int,
) -> dict[str, Any]:
    tool_calls = [trace.model_dump() for trace in result.tool_calls]
    tool_names = [trace["tool"] for trace in tool_calls]
    sources = [source.model_dump() for source in result.sources]
    citations = extract_answer_citations(result.answer)
    evidence_groups = case.get("derived_evidence_groups", [])
    source_refs = [
        {"filename": source["filename"], "page": source.get("page")}
        for source in sources
    ]
    supporting_chunks = supporting_chunks_for_sources(
        sources, chunk_catalog, supporting_context_chars
    )
    evidence_scoring = score_evidence(evidence_groups, source_refs, citations)

    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "reporting_group": case.get("reporting_group", case.get("category", "unknown")),
        "case_source": case.get("case_source", "unknown"),
        "question": case["question"],
        "answerability": case.get("answerability"),
        "requires_web_search": bool(case.get("requires_web_search")),
        "ok": True,
        "error": None,
        "answer": result.answer,
        "reference_answer": case.get("reference_answer"),
        "latency_seconds": latency_seconds,
        "llm_calls": result.llm_calls,
        "stopped_due_to_limit": result.stopped_due_to_limit,
        "tool_calls": tool_calls,
        "tool_selection": score_tool_selection(case["tool_policy"], tool_names),
        "sources": sources,
        "citations": citations,
        "supporting_chunks": supporting_chunks,
        "evidence_scoring": evidence_scoring,
        "faithfulness_scoring": score_faithfulness_proxy(case, evidence_scoring),
        "behavior_scoring": score_behavior(case, result.answer, tool_names, tool_calls),
    }


def serialize_failed_case(
    case: dict[str, Any], exc: Exception, latency_seconds: float) -> dict[str, Any]:
    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "reporting_group": case.get("reporting_group", case.get("category", "unknown")),
        "case_source": case.get("case_source", "unknown"),
        "question": case["question"],
        "answerability": case.get("answerability"),
        "requires_web_search": bool(case.get("requires_web_search")),
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "answer": "",
        "reference_answer": case.get("reference_answer"),
        "latency_seconds": latency_seconds,
        "llm_calls": 0,
        "stopped_due_to_limit": False,
        "tool_calls": [],
        "tool_selection": {
            **case["tool_policy"],
            "selected_tools": [],
            "tool_call_count": 0,
            "expected_tools_satisfied": None,
            "no_unexpected_tools": None,
            "unexpected_tools": [],
        },
        "sources": [],
        "citations": [],
        "supporting_chunks": [],
        "evidence_scoring": score_evidence(case.get("derived_evidence_groups", []), [], []),
        "faithfulness_scoring": {
            "mode": "not_scored",
            "faithful": None,
            "score": None,
            "note": "Agent run failed before faithfulness could be scored.",
        },
        "behavior_scoring": {
            "checks": {},
            "passed": None,
            "failed_tool_call_count": 0,
        },
    }


def score_tool_selection(policy_data: dict[str, Any], tool_names: list[str]) -> dict[str, Any]:
    selected = set(tool_names)
    accepted_plans = policy_data["accepted_tool_plans"]
    if policy_data.get("requires_no_tools"):
        expected_satisfied = not tool_names
    else:
        expected_satisfied = any(set(plan).issubset(selected) for plan in accepted_plans)
    allowed_tools = set(policy_data["allowed_tools"])
    unexpected_tools = [tool for tool in tool_names if tool not in allowed_tools]
    return {
        **policy_data,
        "selected_tools": tool_names,
        "tool_call_count": len(tool_names),
        "expected_tools_satisfied": expected_satisfied,
        "no_unexpected_tools": not unexpected_tools,
        "unexpected_tools": unexpected_tools,
    }


def extract_answer_citations(answer: str) -> list[dict[str, Any]]:
    citations = []
    seen = set()
    for match in CITATION_RE.finditer(answer):
        filename = match.group("filename")
        page = int(match.group("page"))
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"filename": filename, "page": page})
    return citations


def score_evidence(
    evidence_groups: list[dict[str, Any]],
    source_refs: list[dict[str, Any]],
    citation_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    structured_source_score = score_page_refs(evidence_groups, source_refs)
    citation_score = score_page_refs(evidence_groups, citation_refs)
    structured_pages = {
        (ref["filename"], ref.get("page"))
        for ref in source_refs
        if ref.get("page") is not None
    }
    citation_pages_supported = None
    if citation_refs:
        citation_pages_supported = all(
            (ref["filename"], ref.get("page")) in structured_pages
            for ref in citation_refs
        )
    return {
        "source_pages": structured_source_score,
        "answer_citations": citation_score,
        "citation_present": bool(citation_refs),
        "citation_pages_supported_by_structured_sources": citation_pages_supported,
    }


def score_page_refs(
    evidence_groups: list[dict[str, Any]], page_refs: list[dict[str, Any]]
) -> dict[str, Any]:
    if not evidence_groups:
        return {
            "scored": False,
            "page_hit": None,
            "page_recall": None,
            "page_mrr": None,
            "complete_evidence": None,
            "covered_group_ids": [],
            "first_match_position": None,
        }

    covered_group_ids: set[str] = set()
    first_match_position: int | None = None
    for position, ref in enumerate(page_refs, start=1):
        matches = matching_group_ids(ref["filename"], ref.get("page"), evidence_groups)
        if matches and first_match_position is None:
            first_match_position = position
        covered_group_ids.update(matches)

    total_groups = len(evidence_groups)
    page_recall = len(covered_group_ids) / total_groups
    return {
        "scored": True,
        "page_hit": bool(covered_group_ids),
        "page_recall": page_recall,
        "page_mrr": 1.0 / first_match_position if first_match_position else 0.0,
        "complete_evidence": len(covered_group_ids) == total_groups,
        "covered_group_ids": sorted(covered_group_ids),
        "first_match_position": first_match_position,
    }


def score_behavior(
    case: dict[str, Any], answer: str, tool_names: list[str], tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    lower_answer = answer.lower()
    subtype = case.get("subtype")
    answerability = case.get("answerability")
    acceptable_behaviors = set(case.get("acceptable_behaviors", []))
    checks: dict[str, bool] = {}

    if subtype == "metadata_lookup":
        expected_content = case.get("expected_content", [])
        checks["lists_expected_documents"] = all(
            content.lower() in lower_answer for content in expected_content
        )
    if subtype == "invalid_document" or answerability == "invalid_document":
        checks["reports_document_not_found"] = contains_any(lower_answer, NOT_FOUND_TERMS)
    if subtype == "no_tool_needed":
        checks["answers_without_tools"] = not tool_names
    if answerability == "not_reported":
        checks["abstains_when_not_reported"] = contains_any(
            lower_answer, ABSTENTION_TERMS
        )
    if answerability == "external_required":
        checks["acknowledges_external_requirement"] = contains_any(
            lower_answer, EXTERNAL_LIMIT_TERMS
        )
    if "correct_false_premise" in acceptable_behaviors:
        checks["corrects_false_premise"] = contains_any(
            lower_answer, FALSE_PREMISE_TERMS
        )

    answer_scoring = case.get("answer_scoring") or {}
    if "expected_numeric_value" in answer_scoring:
        checks["numeric_answer_correct"] = numeric_answer_matches(
            answer,
            float(answer_scoring["expected_numeric_value"]),
            float(answer_scoring.get("absolute_tolerance", 1e-6)),
        )

    return {
        "checks": checks,
        "passed": all(checks.values()) if checks else None,
        "failed_tool_call_count": sum(1 for tool_call in tool_calls if not tool_call["success"]),
    }


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def numeric_answer_matches(answer: str, expected_value: float, tolerance: float) -> bool:
    for match in NUMBER_RE.finditer(answer):
        try:
            value = float(match.group())
        except ValueError:
            continue
        if abs(value - expected_value) <= tolerance:
            return True
    return False


def score_faithfulness_proxy(
    case: dict[str, Any], evidence_scoring: dict[str, Any]
) -> dict[str, Any]:
    source_score = evidence_scoring["source_pages"]
    if case.get("requires_web_search") or not source_score["scored"]:
        return {
            "mode": "not_scored",
            "faithful": None,
            "score": None,
            "note": "No in-corpus evidence labels are available for faithfulness scoring.",
        }

    citation_supported = evidence_scoring[
        "citation_pages_supported_by_structured_sources"
    ]
    score_parts = [source_score["page_recall"]]
    if evidence_scoring["citation_present"]:
        score_parts.append(evidence_scoring["answer_citations"]["page_recall"])
        if citation_supported is not None:
            score_parts.append(float(citation_supported))

    return {
        "mode": "grounding_proxy",
        "faithful": bool(source_score["page_hit"]) and citation_supported is not False,
        "score": mean_defined(score_parts),
        "note": (
            "Proxy score from labeled source-page coverage and citation support; "
            "use --judge-faithfulness for an additional local LLM judgment."
        ),
    }


def score_faithfulness_with_judge(case_result: dict[str, Any], judge: Any) -> dict[str, Any]:
    supporting_chunks = case_result.get("supporting_chunks", [])
    if not supporting_chunks:
        return {
            "mode": "ollama_judge_not_scored",
            "faithful": None,
            "score": None,
            "note": "No supporting chunks were available to judge against.",
        }

    context = "\n\n".join(
        f"[{chunk['filename']}, page {chunk.get('page')}, chunk {chunk.get('chunk_id')}]\n"
        f"{chunk['text']}"
        for chunk in supporting_chunks
    )
    reference_answer = case_result.get("reference_answer") or "No reference answer."
    prompt = f"""You are evaluating a RAG agent answer.
Use only the SUPPORTING_CONTEXT to judge whether the ANSWER is faithful.
The REFERENCE_ANSWER is only a benchmark hint for relevance; do not require identical wording.
Return only one valid JSON object with these keys:
- faithful: boolean
- answers_question: boolean
- cites_supporting_pages: boolean
- score: number from 0 to 1
- unsupported_claims: array of short strings
- explanation: short string

QUESTION:
{case_result['question']}

REFERENCE_ANSWER:
{reference_answer}

ANSWER:
{case_result['answer']}

SUPPORTING_CONTEXT:
{context}
"""
    raw_response = judge.generate(prompt)
    parsed = parse_json_object(raw_response)
    if not isinstance(parsed, dict):
        return {
            "mode": "ollama_judge_parse_error",
            "faithful": None,
            "score": None,
            "raw_response": raw_response[:2000],
        }

    score = parsed.get("score")
    numeric_score = None
    if isinstance(score, int | float):
        numeric_score = max(0.0, min(1.0, float(score)))
    return {
        "mode": "ollama_judge",
        "faithful": parsed.get("faithful") if isinstance(parsed.get("faithful"), bool) else None,
        "answers_question": parsed.get("answers_question"),
        "cites_supporting_pages": parsed.get("cites_supporting_pages"),
        "score": numeric_score,
        "unsupported_claims": parsed.get("unsupported_claims", []),
        "explanation": parsed.get("explanation"),
        "proxy_score": case_result.get("faithfulness_scoring"),
    }


def parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def aggregate_agent_scores(per_case_results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all_agent_cases": per_case_results,
        "main_non_web": [
            result for result in per_case_results if not result["requires_web_search"]
        ],
        "deferred_web": [
            result for result in per_case_results if result["requires_web_search"]
        ],
    }
    for result in per_case_results:
        groups.setdefault(f"reporting_group:{result['reporting_group']}", []).append(result)
        groups.setdefault(f"category:{result['category']}", []).append(result)

    return {
        group_name: aggregate_one_agent_group(results)
        for group_name, results in groups.items()
        if results
    }


def aggregate_one_agent_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_results = [
        result
        for result in results
        if result["evidence_scoring"]["source_pages"]["scored"]
    ]
    behavior_results = [
        result for result in results if result["behavior_scoring"]["passed"] is not None
    ]
    faithfulness_results = [
        result
        for result in results
        if result["faithfulness_scoring"].get("score") is not None
    ]
    return {
        "case_count": len(results),
        "case_ids": [result["id"] for result in results],
        "run": {
            "ok_rate": mean(result["ok"] for result in results),
            "error_count": sum(1 for result in results if not result["ok"]),
            "avg_latency_seconds": mean(result["latency_seconds"] for result in results),
            "median_latency_seconds": statistics.median(
                result["latency_seconds"] for result in results
            ),
            "avg_llm_calls": mean(result["llm_calls"] for result in results),
            "stopped_due_to_limit_rate": mean(
                result["stopped_due_to_limit"] for result in results
            ),
        },
        "tool_selection": {
            "expected_tools_satisfied_rate": mean_defined(
                result["tool_selection"]["expected_tools_satisfied"]
                for result in results
            ),
            "no_unexpected_tools_rate": mean_defined(
                result["tool_selection"]["no_unexpected_tools"] for result in results
            ),
            "avg_tool_calls": mean(
                result["tool_selection"]["tool_call_count"] for result in results
            ),
        },
        "evidence_and_citations": {
            "scored_case_count": len(evidence_results),
            "structured_source_page_hit_rate": mean_defined(
                result["evidence_scoring"]["source_pages"]["page_hit"]
                for result in evidence_results
            ),
            "structured_source_page_recall": mean_defined(
                result["evidence_scoring"]["source_pages"]["page_recall"]
                for result in evidence_results
            ),
            "structured_source_page_mrr": mean_defined(
                result["evidence_scoring"]["source_pages"]["page_mrr"]
                for result in evidence_results
            ),
            "structured_source_complete_evidence_rate": mean_defined(
                result["evidence_scoring"]["source_pages"]["complete_evidence"]
                for result in evidence_results
            ),
            "citation_present_rate": mean_defined(
                result["evidence_scoring"]["citation_present"]
                for result in evidence_results
            ),
            "citation_page_hit_rate": mean_defined(
                result["evidence_scoring"]["answer_citations"]["page_hit"]
                for result in evidence_results
            ),
            "citation_page_recall": mean_defined(
                result["evidence_scoring"]["answer_citations"]["page_recall"]
                for result in evidence_results
            ),
            "citation_complete_evidence_rate": mean_defined(
                result["evidence_scoring"]["answer_citations"]["complete_evidence"]
                for result in evidence_results
            ),
            "citation_pages_supported_rate": mean_defined(
                result["evidence_scoring"][
                    "citation_pages_supported_by_structured_sources"
                ]
                for result in evidence_results
            ),
        },
        "behavior": {
            "scored_case_count": len(behavior_results),
            "behavior_expectation_pass_rate": mean_defined(
                result["behavior_scoring"]["passed"] for result in behavior_results
            ),
            "failed_tool_call_count": sum(
                result["behavior_scoring"]["failed_tool_call_count"]
                for result in results
            ),
            "check_rates": behavior_check_rates(results),
        },
        "faithfulness": {
            "scored_case_count": len(faithfulness_results),
            "faithful_rate": mean_defined(
                result["faithfulness_scoring"].get("faithful")
                for result in faithfulness_results
            ),
            "faithfulness_score": mean_defined(
                result["faithfulness_scoring"].get("score")
                for result in faithfulness_results
            ),
            "mode_counts": faithfulness_mode_counts(results),
        },
    }


def mean_defined(values: Any) -> float | None:
    defined_values = [float(value) for value in values if value is not None]
    if not defined_values:
        return None
    return sum(defined_values) / len(defined_values)


def faithfulness_mode_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        mode = result["faithfulness_scoring"].get("mode", "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def behavior_check_rates(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values_by_check: dict[str, list[bool]] = {}
    for result in results:
        for check_name, check_value in result["behavior_scoring"]["checks"].items():
            values_by_check.setdefault(check_name, []).append(check_value)
    return {
        check_name: {
            "case_count": len(values),
            "pass_rate": mean(values),
        }
        for check_name, values in sorted(values_by_check.items())
    }


def summarize_agent_latency(
    case_latencies: list[float], *, indexing_seconds: float, agent_seconds: float
) -> dict[str, Any]:
    return {
        "indexing_total": indexing_seconds,
        "agent_total": agent_seconds,
        "case_count": len(case_latencies),
        "case_mean": mean(case_latencies),
        "case_median": statistics.median(case_latencies) if case_latencies else 0.0,
        "case_min": min(case_latencies) if case_latencies else 0.0,
        "case_max": max(case_latencies) if case_latencies else 0.0,
    }


def build_agent_run_metadata(
    *,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    corpus_files: list[CorpusFile],
    cases: list[dict[str, Any]],
    indexed_documents: list[dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
    latency: dict[str, Any],
    project_settings: Any,
    resource_measurements: dict[str, Any],
    allow_hash_mismatch: bool,
    include_deferred_web: bool,
    explicit_case_ids: tuple[str, ...] | None,
    judge_faithfulness: bool,
    supporting_context_chars: int,
) -> dict[str, Any]:
    from app.agent.prompts import AGENT_SYSTEM_PROMPT

    corpus_hashes = [corpus_file.to_json() for corpus_file in corpus_files]
    return {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("version"),
        "benchmark_sha256": sha256_file(benchmark_path),
        "evaluation_type": "agent_loop",
        "run_started_at": started_at.isoformat(),
        "run_finished_at": finished_at.isoformat(),
        "corpus_hash_algorithm": "sha256",
        "corpus_sha256": combined_corpus_sha256(corpus_files),
        "corpus_files": corpus_hashes,
        "corpus_hashes_verified": all(item["hash_matches"] for item in corpus_hashes),
        "allow_hash_mismatch": allow_hash_mismatch,
        "case_selection": {
            "case_ids": [case["id"] for case in cases],
            "explicit_case_ids": list(explicit_case_ids) if explicit_case_ids else None,
            "include_deferred_web": include_deferred_web,
            "deferred_web_cases_included": [
                case["id"] for case in cases if case.get("requires_web_search")
            ],
            "note": (
                "Q28-Q30 require web search and are excluded unless "
                "--include-deferred-web is set. Without a web tool, they only "
                "measure capability-limit behavior."
            ),
        },
        "embedding_model_name": project_settings.embedding_model_name,
        "embedding_model_revision": "unknown",
        "chunk_size": project_settings.chunk_size,
        "chunk_overlap": project_settings.chunk_overlap,
        "agent_configuration": {
            "llm_provider": "ollama",
            "llm_model": project_settings.llm_model,
            "ollama_host": project_settings.ollama_host,
            "max_agent_steps": project_settings.max_agent_steps,
            "summary_batch_chars": project_settings.summary_batch_chars,
            "max_summary_batches": project_settings.max_summary_batches,
            "available_tools": list(AVAILABLE_AGENT_TOOLS),
            "system_prompt_sha256": sha256_text(AGENT_SYSTEM_PROMPT),
            "judge_faithfulness": judge_faithfulness,
            "supporting_context_chars": supporting_context_chars,
        },
        "scoring_configuration": {
            "tool_selection": (
                "Case-specific required/allowed tool sets are derived from "
                "benchmark behavior labels and requires_calculation flags."
            ),
            "faithfulness": (
                "Default scoring is a deterministic grounding proxy from source "
                "and citation overlap. With --judge-faithfulness, the configured "
                "local Ollama model also judges the answer against saved "
                "supporting chunks."
            ),
            "citation_unit": "filename_and_one_based_pdf_page",
            "behavior_checks": (
                "Deterministic keyword/numeric checks for abstention, invalid "
                "documents, false-premise correction, metadata listing, direct "
                "answers, web-capability limits, and simple arithmetic."
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


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_agent_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_metadata.json", payload["metadata"])
    write_json(output_dir / "aggregate_scores.json", payload["aggregate_scores"])
    write_json(output_dir / "per_case_results.json", payload["per_case_results"])
    with (output_dir / "tool_traces.jsonl").open("w", encoding="utf-8") as file:
        for case_result in payload["per_case_results"]:
            for index, tool_call in enumerate(case_result["tool_calls"], start=1):
                file.write(
                    json.dumps(
                        {
                            "case_id": case_result["id"],
                            "category": case_result["category"],
                            "reporting_group": case_result["reporting_group"],
                            "tool_call_index": index,
                            **tool_call,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as file:
        for case_result in payload["per_case_results"]:
            file.write(
                json.dumps(
                    {
                        "case_id": case_result["id"],
                        "category": case_result["category"],
                        "question": case_result["question"],
                        "answer": case_result["answer"],
                        "citations": case_result["citations"],
                        "sources": case_result["sources"],
                        "faithfulness_scoring": case_result["faithfulness_scoring"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
