#!/usr/bin/env python3
"""
demo_cache_performance.py

Sends a curated set of queries -- originals, exact repeats, and semantic
paraphrases -- to the running LMS Semantic Cache API. Records cache hit/miss
and latency for every call, then generates a plain-text performance report.

This is a STANDALONE script, not a pytest test. Run it directly:

    python demo_cache_performance.py
    python demo_cache_performance.py --base-url http://localhost:8000
    python demo_cache_performance.py --dataset demo_queries.json
    python demo_cache_performance.py --output my_report.txt
    python demo_cache_performance.py --delay 0.5

Prerequisites
-------------
- The API server must already be running:  python main.py
- Milvus + Redis must be up:                docker compose up -d
- Courses referenced in the dataset will be auto-registered if missing.

Dataset
-------
The query set lives in demo_queries.json (same directory as this script by
default). It is NOT hardcoded here -- edit the JSON file to add, remove, or
reword queries without touching this script.

Each entry has:
  group        - concept label, groups related queries in the report
  course_tag   - which course partition to query
  query        - the actual question text
  kind         - "original" | "repeat" | "paraphrase"
  expected_hit - engineered prediction, used to score design accuracy

Dataset design note (see demo_queries.json for the live numbers)
------------------------------------------------------------------
Every concept group follows: original -> repeat -> paraphrase

  - original   - always a cold MISS (first time this concept is asked)
  - repeat     - identical text to the original - always a HIT, sim~1.0
  - paraphrase - reworded version of the original, two designs:
      TIGHT  - close rewording, varied phrasing -> expected to clear the
               0.92 similarity threshold -> HIT
      LOOSE  - same concept, very different wording -> expected to fall
               below the threshold -> MISS

This is an ENGINEERED TARGET based on expected embedding behaviour, not a
guarantee. The actual embedding model may score individual pairs slightly
differently. The report shows both the PREDICTED outcome (per query) and
the ACTUAL outcome side by side, so any drift from the target is visible
rather than hidden.

Output
------
- Live console output as each query runs
- A .txt report with hit rate, latency stats, prediction accuracy, and a
  per-concept breakdown
- A .json file with the raw results for further analysis
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

@dataclass
class QueryCase:
    group:        str    # concept label, groups related queries in the report
    course_tag:   str
    query:        str
    kind:         str    # "original" | "repeat" | "paraphrase"
    expected_hit: bool   # engineered prediction - used to score design accuracy


def load_query_dataset(path: Path) -> list[QueryCase]:
    """
    Load and validate the query dataset from a JSON file.

    Expected shape:
        {
          "description": "...",
          "queries": [
            {"group": "...", "course_tag": "...", "query": "...",
             "kind": "original|repeat|paraphrase", "expected_hit": true|false},
            ...
          ]
        }
    """
    if not path.exists():
        print(f"ERROR: Dataset file not found: {path}")
        print("       Make sure demo_queries.json is in the same directory as this")
        print("       script, or pass a custom path with --dataset <path>.")
        sys.exit(1)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse {path} as JSON: {exc}")
        sys.exit(1)

    entries = raw.get("queries")
    if not isinstance(entries, list) or not entries:
        print(f"ERROR: {path} has no 'queries' list, or it is empty.")
        sys.exit(1)

    required_fields = {"group", "course_tag", "query", "kind", "expected_hit"}
    cases: list[QueryCase] = []
    for idx, entry in enumerate(entries, 1):
        missing = required_fields - entry.keys()
        if missing:
            print(f"ERROR: Entry #{idx} in {path} is missing fields: {missing}")
            sys.exit(1)
        if entry["kind"] not in ("original", "repeat", "paraphrase"):
            print(f"ERROR: Entry #{idx} has invalid kind '{entry['kind']}'. "
                  f"Must be 'original', 'repeat', or 'paraphrase'.")
            sys.exit(1)
        cases.append(QueryCase(
            group=entry["group"],
            course_tag=entry["course_tag"],
            query=entry["query"],
            kind=entry["kind"],
            expected_hit=bool(entry["expected_hit"]),
        ))

    return cases


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class QueryResult:
    case:           QueryCase
    cache_hit:      bool
    similarity:     Optional[float]
    latency_ms:     float
    answer_preview: str
    error:          Optional[str] = None
    http_status:    int = 200

    @property
    def prediction_correct(self) -> bool:
        return self.error is None and self.cache_hit == self.case.expected_hit


# -----------------------------------------------------------------------------
# HTTP calls
# -----------------------------------------------------------------------------

def ensure_courses_registered(client: httpx.Client, base_url: str, courses: list[str]) -> None:
    """Idempotently register all courses used in the dataset before running."""
    print("Ensuring required courses are registered...")
    for tag in courses:
        try:
            resp = client.post(f"{base_url}/api/v1/courses", json={"course_tag": tag}, timeout=15.0)
            status = "created" if resp.status_code == 201 else "already exists"
            print(f"  - {tag:<10} -> {status}")
        except httpx.RequestError as exc:
            print(f"  - {tag:<10} -> WARNING could not register: {exc}")
    print()


def run_query(client: httpx.Client, base_url: str, case: QueryCase) -> QueryResult:
    payload = {"query": case.query, "course_tag": case.course_tag}
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/api/v1/query", json=payload, timeout=90.0)
        wall_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            return QueryResult(
                case=case, cache_hit=False, similarity=None,
                latency_ms=wall_ms, answer_preview="",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )

        body = resp.json()
        answer = body.get("answer", "")
        preview = (answer[:150] + "...") if len(answer) > 150 else answer

        return QueryResult(
            case=case,
            cache_hit=body.get("cache_hit", False),
            similarity=body.get("similarity"),
            latency_ms=body.get("latency_ms", wall_ms),
            answer_preview=preview,
        )
    except httpx.RequestError as exc:
        wall_ms = (time.perf_counter() - t0) * 1000
        return QueryResult(
            case=case, cache_hit=False, similarity=None,
            latency_ms=wall_ms, answer_preview="", error=str(exc),
        )


def print_live(idx: int, total: int, result: QueryResult) -> None:
    if result.error:
        icon = "ERR "
    elif result.cache_hit:
        icon = "HIT "
    else:
        icon = "MISS"

    sim_str = f"{result.similarity:.4f}" if result.similarity is not None else "  N/A "
    pred    = "OK" if result.prediction_correct else ("XX" if not result.error else "  ")

    print(
        f"[{idx:>2}/{total}] {icon} | pred:{pred} | {result.latency_ms:>9.1f}ms | sim={sim_str} | "
        f"{result.case.course_tag:<8} | {result.case.kind:<10} | {result.case.query[:50]}"
    )
    if result.error:
        print(f"          -> {result.error}")


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def generate_report(
    results: list[QueryResult],
    base_url: str,
    predicted_rate: float,
    total_groups: int,
) -> str:
    total      = len(results)
    successful = [r for r in results if r.error is None]
    errors     = [r for r in results if r.error is not None]
    hits       = [r for r in successful if r.cache_hit]
    misses     = [r for r in successful if not r.cache_hit]

    actual_hit_rate = (len(hits) / len(successful) * 100) if successful else 0.0

    avg_hit_latency  = sum(r.latency_ms for r in hits) / len(hits) if hits else 0.0
    avg_miss_latency = sum(r.latency_ms for r in misses) / len(misses) if misses else 0.0
    speedup          = (avg_miss_latency / avg_hit_latency) if avg_hit_latency > 0 else 0.0
    avg_overall      = (sum(r.latency_ms for r in successful) / len(successful)) if successful else 0.0

    predictable     = successful
    correct_preds   = [r for r in predictable if r.prediction_correct]
    prediction_acc  = (len(correct_preds) / len(predictable) * 100) if predictable else 0.0

    paraphrases       = [r for r in successful if r.case.kind == "paraphrase"]
    paraphrase_hits   = [r for r in paraphrases if r.cache_hit]
    paraphrase_rate   = (len(paraphrase_hits) / len(paraphrases) * 100) if paraphrases else 0.0

    tight_paraphrases = [r for r in paraphrases if r.case.expected_hit]
    loose_paraphrases = [r for r in paraphrases if not r.case.expected_hit]
    tight_hit_rate = (sum(1 for r in tight_paraphrases if r.cache_hit) / len(tight_paraphrases) * 100) if tight_paraphrases else 0.0
    loose_hit_rate = (sum(1 for r in loose_paraphrases if r.cache_hit) / len(loose_paraphrases) * 100) if loose_paraphrases else 0.0

    lines: list[str] = []
    lines.append("=" * 82)
    lines.append("  LMS SEMANTIC CACHE - PERFORMANCE REPORT")
    lines.append("=" * 82)
    lines.append(f"  Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Target API     : {base_url}")
    lines.append(f"  Total queries  : {total}  ({total_groups} concept groups)")
    lines.append("")

    lines.append("-" * 82)
    lines.append("  SUMMARY")
    lines.append("-" * 82)
    lines.append(f"  Successful requests     : {len(successful)} / {total}")
    lines.append(f"  Failed requests          : {len(errors)}")
    lines.append(f"  Cache HITs               : {len(hits)}")
    lines.append(f"  Cache MISSes             : {len(misses)}")
    lines.append(f"  Predicted hit rate       : {predicted_rate:.1f}%  (from dataset design)")
    lines.append(f"  ACTUAL hit rate          : {actual_hit_rate:.1f}%")
    lines.append(f"  Prediction accuracy      : {prediction_acc:.1f}%  "
                 f"({len(correct_preds)}/{len(predictable)} queries matched the engineered design)")
    lines.append("")
    lines.append(f"  Avg latency (HIT)        : {avg_hit_latency:>10.2f} ms")
    lines.append(f"  Avg latency (MISS)       : {avg_miss_latency:>10.2f} ms")
    lines.append(f"  Avg latency (overall)    : {avg_overall:>10.2f} ms")
    lines.append(f"  Cache speedup factor     : {speedup:>10.2f}x faster on a hit")
    lines.append("")

    lines.append("-" * 82)
    lines.append("  SEMANTIC SEARCH EFFECTIVENESS")
    lines.append("-" * 82)
    lines.append(f"  Total paraphrases tested        : {len(paraphrases)}")
    lines.append(f"  Paraphrases that hit cache       : {len(paraphrase_hits)} ({paraphrase_rate:.1f}%)")
    lines.append("")
    lines.append(f"  TIGHT paraphrases  : {len(tight_paraphrases)} tested, "
                 f"{tight_hit_rate:.1f}% hit rate  (designed to HIT)")
    lines.append(f"  LOOSE paraphrases  : {len(loose_paraphrases)} tested, "
                 f"{loose_hit_rate:.1f}% hit rate  (designed to MISS)")
    lines.append("")
    lines.append("  -> TIGHT paraphrases hitting near 100% and LOOSE paraphrases hitting")
    lines.append("     near 0% confirms the embedding model is drawing a sensible semantic")
    lines.append("     boundary at the similarity threshold - not too loose, not too strict.")
    lines.append("")

    lines.append("-" * 82)
    lines.append("  PER-CONCEPT-GROUP BREAKDOWN")
    lines.append("-" * 82)

    groups: dict[str, list[QueryResult]] = {}
    for r in successful:
        groups.setdefault(r.case.group, []).append(r)

    for group_name, group_results in groups.items():
        course = group_results[0].case.course_tag
        design = "tight" if any(r.case.kind == "paraphrase" and r.case.expected_hit for r in group_results) else "loose"
        lines.append(f"\n  > {group_name}  ({course})  [{design} paraphrase design]")
        for r in group_results:
            icon = "HIT " if r.cache_hit else "MISS"
            mark = "OK" if r.prediction_correct else "XX"
            sim  = f"sim={r.similarity:.4f}" if r.similarity is not None else "sim=N/A   "
            lines.append(
                f"      [{r.case.kind:<10}] {icon} {mark} | {sim} | {r.latency_ms:>9.1f}ms | \"{r.case.query}\""
            )

    if errors:
        lines.append("")
        lines.append("-" * 82)
        lines.append("  ERRORS")
        lines.append("-" * 82)
        for r in errors:
            lines.append(f"  [{r.case.course_tag}] \"{r.case.query}\" -> {r.error}")

    lines.append("")
    lines.append("-" * 82)
    lines.append("  FULL QUERY LOG")
    lines.append("-" * 82)
    lines.append(f"  {'#':<3} {'Course':<9} {'Kind':<11} {'Hit':<5} {'Pred':<5} {'Sim':<8} {'Latency(ms)':<13} Query")
    lines.append("  " + "-" * 78)
    for idx, r in enumerate(results, 1):
        sim  = f"{r.similarity:.4f}" if r.similarity is not None else "N/A"
        hit  = "YES" if r.cache_hit else ("ERR" if r.error else "NO")
        pred = "OK" if r.prediction_correct else ("." if r.error else "XX")
        lines.append(
            f"  {idx:<3} {r.case.course_tag:<9} {r.case.kind:<11} {hit:<5} {pred:<5} {sim:<8} "
            f"{r.latency_ms:<13.1f} {r.case.query}"
        )

    lines.append("")
    lines.append("=" * 82)
    lines.append("  END OF REPORT")
    lines.append("=" * 82)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run demo queries against the LMS Semantic Cache API and generate a performance report."
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the query dataset JSON file. Defaults to demo_queries.json next to this script.",
    )
    parser.add_argument("--output", default=None, help="Report .txt output path")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between requests, in seconds")
    parser.add_argument("--skip-course-setup", action="store_true", help="Skip auto-registering courses")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else (Path(__file__).parent / "demo_queries.json")
    query_dataset = load_query_dataset(dataset_path)

    required_courses = sorted({c.course_tag for c in query_dataset})
    total_groups      = len(set(c.group for c in query_dataset))
    expected_hits      = sum(1 for c in query_dataset if c.expected_hit)
    predicted_rate     = (expected_hits / len(query_dataset) * 100) if query_dataset else 0.0

    print(f"\nDataset file      : {dataset_path}")
    print(f"Target API        : {args.base_url}")
    print(f"Total queries     : {len(query_dataset)}  ({total_groups} concept groups)")
    print(f"Predicted hit rate: {predicted_rate:.1f}%")
    print(f"Courses used      : {', '.join(required_courses)}\n")
    print("-" * 82)

    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/health", timeout=10.0)
            if health.status_code != 200:
                print(f"WARNING: /health returned {health.status_code}. Continuing anyway...\n")
            else:
                print("API is healthy.\n")
    except httpx.RequestError as exc:
        print(f"ERROR: Cannot reach API at {args.base_url}: {exc}")
        print("   Make sure the server is running:  python main.py")
        sys.exit(1)

    results: list[QueryResult] = []

    with httpx.Client() as client:
        if not args.skip_course_setup:
            ensure_courses_registered(client, args.base_url, required_courses)

        print("-" * 82)
        for idx, case in enumerate(query_dataset, 1):
            result = run_query(client, args.base_url, case)
            results.append(result)
            print_live(idx, len(query_dataset), result)
            time.sleep(args.delay)

    print("-" * 82)
    print("\nGenerating report...\n")

    report_text = generate_report(results, args.base_url, predicted_rate, total_groups)
    print(report_text)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"cache_performance_report_{timestamp}.txt"
    Path(output_path).write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to : {output_path}")

    json_path = str(Path(output_path).with_suffix(".json"))
    raw_data = [
        {
            "group":        r.case.group,
            "course_tag":   r.case.course_tag,
            "query":        r.case.query,
            "kind":         r.case.kind,
            "expected_hit": r.case.expected_hit,
            "cache_hit":    r.cache_hit,
            "prediction_correct": r.prediction_correct,
            "similarity":   r.similarity,
            "latency_ms":   r.latency_ms,
            "error":        r.error,
        }
        for r in results
    ]
    Path(json_path).write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
    print(f"Raw data saved to: {json_path}\n")


if __name__ == "__main__":
    main()
