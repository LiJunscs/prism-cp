"""Read frozen evidence and summarize all-rank complete training updates."""

import gzip
import json


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def summarize_records(folder, manifest, total_steps, measured_steps, expected_batches):
    compact = folder / "steps_by_rank.json.gz"
    if compact.exists():
        with gzip.open(compact, "rt", encoding="utf-8") as handle:
            saved = json.load(handle)
        ranks = [saved.get(str(rank), []) for rank in range(16)]
    else:
        ranks = []
        for rank in range(16):
            path = folder / f"steps_rank{rank}.jsonl"
            ranks.append([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                         if path.exists() else [])
    flat = [row for rows in ranks for row in rows]
    result = dict(source=str(folder), records=len(flat),
                  steps_by_rank={rank: [r["step"] for r in rows] for rank, rows in enumerate(ranks)})
    if not flat:
        return dict(result, status="no_complete_update_records")
    result["peak_bytes"] = {key: max(r[key] for r in flat) for key in
                            ("peak_allocated", "peak_reserved", "memory_required_bytes")}
    result["peak_GiB"] = {key: value/2**30 for key, value in result["peak_bytes"].items()}
    result["minimum_memory_budget_bytes"] = min(r["memory_budget_bytes"] for r in flat)
    checks = dict(
        input_tokens=all(r["tokens"] == manifest["input_tokens"] for r in flat),
        effective_loss_tokens=all(r["effective_loss_tokens"] == manifest["loss_tokens"] for r in flat),
        full_update=all(r["optimizer_updates"] == 1 and r["skipped_updates"] == 0 for r in flat),
        finite=all(r["finite"] for r in flat),
        parameters_synchronized=all(r["parameter_sync_completed"] for r in flat),
        forward_backward=all(r["execution_microbatches_per_update"] == expected_batches
            and r["packet_fb"]["forwards"] == r["packet_fb"]["backwards"] == expected_batches for r in flat),
        physical_memory=all(r["all_ranks_memory_admitted"] for r in flat))
    result["checks"] = checks
    if not all(value for key, value in checks.items() if key != "physical_memory"):
        return dict(result, status="training_contract_failed")
    if not checks["physical_memory"]:
        return dict(result, status="actual_memory_guard_rejected",
                    first_rejected_step=min(r["step"] for r in flat if not r["all_ranks_memory_admitted"]))
    if any([r["step"] for r in rows] != list(range(1, total_steps+1)) for rows in ranks):
        return dict(result, status="incomplete")
    times = [max(rows[step-1]["elapsed_ms"] for rows in ranks)/1000 for step in measured_steps]
    return dict(result, status="complete", measured_steps=list(measured_steps),
                rankmax_seconds=times, mean_seconds=sum(times)/len(times))
