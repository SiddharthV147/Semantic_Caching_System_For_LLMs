import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

# ── Step 0: Tear down ALL existing data (clean slate) ────────────────────────
# This is the root cause of the "Test 1 fails on re-run" bug.
# Without this, Milvus and Redis still hold vectors/keys from the previous run,
# so the very first query in the new run is a cache HIT — not a cold MISS.
print("\n" + "═" * 55)
print("  STEP 0 — Resetting databases (clean slate)")
print("═" * 55)

from src.database.db_setup import teardown_all_data, setup_all_databases

teardown_all_data()

# ── Step 1: Recreate collections, indexes, and partitions ────────────────────
print("\n" + "═" * 55)
print("  STEP 1 — Rebuilding schema and indexes")
print("═" * 55)

setup_all_databases(initial_course_tags=["CS101", "MATH202", "ENG301"])

# ── Test helpers ─────────────────────────────────────────────────────────────
from src.orchestrator import process_lms_query


def fmt_sim(val) -> str:
    return f"{val:.4f}" if val is not None else "N/A"


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


QUERY    = "What is gradient descent?"
COURSE_A = "CS101"
COURSE_B = "MATH202"

# ── TEST 1: Cold query — must be a MISS ──────────────────────────────────────
section("TEST 1 — Expected: Cache MISS  (cold query, empty cache)")
r1 = process_lms_query(user_query=QUERY, course_tag=COURSE_A)

print(f"  Cache Hit  : {r1.cache_hit}")
print(f"  Similarity : {fmt_sim(r1.similarity)}")
print(f"  KB chunks  : {r1.kb_chunks_used}")
print(f"  Latency    : {r1.latency_ms:.1f} ms")
print(f"  Error      : {r1.error}")
print(f"  Answer     : {r1.answer[:100]}…")

assert not r1.cache_hit, f"TEST 1 FAILED — expected MISS, got HIT (sim={fmt_sim(r1.similarity)})"
assert r1.error is None, f"TEST 1 FAILED — unexpected error: {r1.error}"
print("  ✅ TEST 1 PASSED")


# ── TEST 2: Same query — must be a HIT with sim ≥ threshold ──────────────────
section("TEST 2 — Expected: Cache HIT  (identical query, sim = 1.0)")
r2 = process_lms_query(user_query=QUERY, course_tag=COURSE_A)

print(f"  Cache Hit  : {r2.cache_hit}")
print(f"  Similarity : {fmt_sim(r2.similarity)}")
print(f"  Latency    : {r2.latency_ms:.1f} ms")
print(f"  Error      : {r2.error}")
print(f"  Answer     : {r2.answer[:100]}…")

assert r2.cache_hit, \
    f"TEST 2 FAILED — identical query should be a HIT (got MISS, best_score={fmt_sim(r2.similarity)})"
assert r2.similarity >= 0.92, \
    f"TEST 2 FAILED — similarity {fmt_sim(r2.similarity)} is below threshold 0.92"
assert r2.error is None, f"TEST 2 FAILED — unexpected error: {r2.error}"
print("  ✅ TEST 2 PASSED")


# ── TEST 3: Same question, different course — must be a MISS ─────────────────
section("TEST 3 — Expected: Cache MISS  (course isolation — MATH202 ≠ CS101)")
r3 = process_lms_query(user_query=QUERY, course_tag=COURSE_B)

print(f"  Cache Hit  : {r3.cache_hit}")
print(f"  Similarity : {fmt_sim(r3.similarity)}")
print(f"  Latency    : {r3.latency_ms:.1f} ms")
print(f"  Error      : {r3.error}")

assert not r3.cache_hit, (
    f"TEST 3 FAILED — cross-course cache hit detected! "
    f"CS101 vectors leaked into MATH202 search (sim={fmt_sim(r3.similarity)}). "
    "Course isolation is broken."
)
print("  ✅ TEST 3 PASSED — MATH202 correctly isolated from CS101 cache")


# ── TEST 4: Hit must be measurably faster than a warm miss ───────────────────
section("TEST 4 — Expected: Hit latency < Miss latency")

# r3 is the warm-miss baseline: model already loaded, Milvus warm, no KB data
# r2 is the cache hit
miss_ms = r3.latency_ms
hit_ms  = r2.latency_ms

print(f"  Warm miss latency  : {miss_ms:.1f} ms  (r3 — MATH202 miss, warm model)")
print(f"  Cache hit latency  : {hit_ms:.1f} ms  (r2 — CS101 hit)")
print(f"  Speedup            : {miss_ms / hit_ms:.1f}x")

assert hit_ms < miss_ms, (
    f"TEST 4 FAILED — hit ({hit_ms:.1f}ms) was not faster than miss ({miss_ms:.1f}ms). "
    "Check flush() overhead in update_cache()."
)
print("  ✅ TEST 4 PASSED")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  ALL 4 TESTS PASSED ✅")
print("═" * 55)
print(f"  Cold miss (incl. model load) : {r1.latency_ms:.1f} ms")
print(f"  Warm miss (MATH202)          : {r3.latency_ms:.1f} ms")
print(f"  Cache hit (CS101)            : {r2.latency_ms:.1f} ms")
print(f"  Cache speedup vs warm miss   : {r3.latency_ms / r2.latency_ms:.1f}x\n")
