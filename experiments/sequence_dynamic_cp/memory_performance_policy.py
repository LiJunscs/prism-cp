"""Apply a shared memory/throughput preference to saved N and joint candidates.

CPU arithmetic only: no model evaluation, native replay or GPU launch. The
fixed reference for each input prevents compounding the allowed memory growth.
Physical feasibility is independent of this configurable preference.
"""

import math


def tradeoff(reference_time, reference_peak, candidate_time, candidate_peak,
             *, min_gain_per_memory_growth=2.0, fits=True):
    values = (reference_time, reference_peak, candidate_time, candidate_peak)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Times and memory peaks must be finite and positive")
    if not math.isfinite(min_gain_per_memory_growth) or min_gain_per_memory_growth < 0:
        raise ValueError("The gain multiplier must be finite and nonnegative")
    growth = candidate_peak / reference_peak - 1
    gain = reference_time / candidate_time - 1
    required = min_gain_per_memory_growth * max(0., growth)
    # This gate concerns buying speed with extra memory. Selection separately
    # keeps the reference available, so slower low-memory candidates cannot win.
    worth_growth = growth <= 0 or gain >= required or math.isclose(gain, required, abs_tol=1e-12)
    reason = ("physical_memory_rejected" if not fits else
              "insufficient_throughput_gain" if not worth_growth else
              "no_memory_increase" if growth <= 0 else "gain_covers_memory_growth")
    return dict(accepted=bool(fits and worth_growth), reason=reason,
                memory_growth_pct=100*growth, throughput_gain_pct=100*gain,
                required_throughput_gain_pct=100*required)


def select(candidates, reference_id, multiplier=2.0):
    reference = next(c for c in candidates if c["id"] == reference_id)
    if not reference["fits"]:
        raise ValueError("The common reference must be physically feasible")
    rows = []
    for candidate in candidates:
        decision = tradeoff(reference["time"], reference["peak"], candidate["time"],
                            candidate["peak"], min_gain_per_memory_growth=multiplier,
                            fits=candidate["fits"])
        rows.append(dict(candidate, decision=decision))
    accepted = [c for c in rows if c["decision"]["accepted"]]
    key = lambda c: (c["time"], c["peak"], c["id"] != reference_id, c["id"])
    best_native = min((c for c in accepted if c["kind"] == "native"), key=key)
    selected = min(accepted, key=key)
    # Preserve the raw feasible frontier as evidence independent of preference.
    feasible = [c for c in rows if c["fits"]]
    frontier = [c["id"] for c in sorted(feasible, key=lambda c: (c["peak"], c["time"]))
                if not any(d["peak"] <= c["peak"] and d["time"] <= c["time"]
                           and (d["peak"] < c["peak"] or d["time"] < c["time"])
                           for d in feasible)]
    return dict(reference=reference_id, candidates=rows, best_native=best_native["id"],
                best_native_time=best_native["time"], selected=selected["id"],
                selected_time=selected["time"], physical_time_memory_frontier=frontier)
