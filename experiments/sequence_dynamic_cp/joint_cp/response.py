"""Hardware-specific responses; no fitting to scheduler outcomes.

Legacy H800 residuals retain their original transfer policies. H100 uses separate
full-module GQA/GDN anchors and explicitly declared work/CP extrapolations.
Both include operator communication; MLP, routing and memory remain separate.
"""

import csv
from copy import deepcopy
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H100_SCHEMA = "h100_operator_anchors_v1"


def operator_work(model, symbol, tokens, squares):
    """Declared useful F+B FLOPs; excludes routing and implementation overhead."""
    h, d = model.hidden, model.gdn_heads * model.gdn_dim
    if symbol == "*":
        return (6 * (h * (model.q_heads + 2 * model.kv_heads) * model.head_dim + h*h) * tokens
                + 6 * model.q_heads * model.head_dim * squares)
    if symbol == "G":
        return 6 * (6*h*d + 4*model.gdn_heads*model.gdn_dim**2) * tokens
    if symbol == "M":
        return 18 * h * model.ffn * tokens
    raise ValueError("Unknown operator")


def validate_h100_response(profile, model):
    shape = (model.hidden, model.q_heads, model.kv_heads, model.head_dim,
             model.gdn_heads, model.gdn_dim, model.ranks_per_node)
    if shape != (4096, 32, 8, 128, 16, 128, 8):
        raise ValueError("H100 anchors require H4096/Q32/KV8/D128/GDN16x128 and 8 GPUs per node")
    if profile.get("hardware") != "H100" or profile.get("anchor_tokens") != 262144:
        raise ValueError("Unknown H100 anchor hardware or length")
    if (profile.get("shape_policy") != "useful_work_scaling_unvalidated"
            or profile.get("unmeasured_cp_policy") != "constant_cp8_per_gpu_efficiency"):
        raise ValueError("H100 extrapolation must have an explicit supported policy")
    anchors = profile.get("anchors", {})
    if set(anchors) != {"GQA", "GDN"}:
        raise ValueError("H100 requires separate GQA and GDN anchors")
    domain = set(anchors["GQA"]["milliseconds"])
    if domain not in ({"8", "16"}, {"8", "16", "32"}):
        raise ValueError("H100 measured CP domain must be 8/16 or 8/16/32")
    if "32" in domain and not profile.get("cp32_source"):
        raise ValueError("H100 CP32 requires its independent measured source")
    for anchor in anchors.values():
        if set(anchor["milliseconds"]) != domain:
            raise ValueError("H100 GQA/GDN measured CP domains must match")
        if any(not math.isfinite(t) or t <= 0 for t in anchor["milliseconds"].values()):
            raise ValueError("H100 anchor times must be finite and positive")


def load_response(regime="joint", cp16="paired_linear", gdn=True, cp32=None, gdn32_multiplier=1.):
    if regime not in ("joint", "single_b1", "fixed_g8"):
        raise ValueError("Unknown frozen fit regime")
    if cp16 not in ("carry8", "paired_linear", "paired_quadratic"):
        raise ValueError("Unknown CP16 response policy")
    if cp32 not in (None, "work_scaling", "carry_residual", "double_residual"):
        raise ValueError("Unknown CP32 hypothesis")
    source = ROOT / "results/historical_cp_response_20261012/profiles" / f"{regime}.json"
    profile = json.loads(source.read_text())
    pairs_path = ROOT / "results/cross_node_cp_evidence_20261012/paired_response.csv"
    with pairs_path.open() as handle:
        pairs = [dict(length=int(r["length"]), ratio=float(r["CP16_over_CP8"]),
                      cp8_ms=float(r["CP8_FB_ms"]), cp16_ms=float(r["CP16_FB_ms"]))
                 for r in csv.DictReader(handle)]
    return dict(schema="joint_cp_response_v1", name=f"{regime}_{cp16}" + ("" if gdn else "_gqa_only")
                + (f"_cp32_{cp32}_gdn{gdn32_multiplier:g}" if cp32 else ""),
                fits=deepcopy(profile["fits"] if gdn else {"GQA": profile["fits"]["GQA"]}),
                cp16_policy=cp16, paired_anchors=pairs,
                source=str(source.relative_to(ROOT)), paired_source=str(pairs_path.relative_to(ROOT)),
                scope=profile["scope"], gdn_transfer=profile["gdn_transfer"] if gdn else "not transferred",
                packing_extrapolation=profile["packing_extrapolation"],
                cp1_policy="retain current model FLOP coefficients; no historical absolute CP1 time transfer",
                cp16_gdn_policy="CP8 prior plus nonnegative analytic inter-minus-intra communication; unmeasured",
                cp16_gqa_policy="paired 64/256Ki B1 ratios; packed extension is unvalidated" if cp16 != "carry8"
                                else "CP8 prior plus nonnegative analytic topology increment; no paired ratio",
                cp32_policy=cp32, gdn32_multiplier=gdn32_multiplier,
                cp32_scope="No CP32 measurements. GQA residual hypothesis from CP16; GDN CP8 fractions plus modeled topology, times stated sensitivity multiplier. Four 8-GPU nodes; link/route model unvalidated.",
                accounting="replace calibrated operator communication and latency; retain layout traffic and memory")


def validate_response(profile, model):
    if profile.get("schema") == H100_SCHEMA:
        validate_h100_response(profile, model)
        return
    if profile.get("schema") != "joint_cp_response_v1":
        raise ValueError("Unknown CP response schema")
    if profile.get("cp16_policy") not in ("carry8", "paired_linear", "paired_quadratic"):
        raise ValueError("Unknown CP16 response policy")
    if profile.get("cp32_policy") not in (None, "work_scaling", "carry_residual", "double_residual"):
        raise ValueError("Unknown CP32 hypothesis")
    multiplier = profile.get("gdn32_multiplier", 1.)
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("CP32 GDN stress multiplier must be finite and >=1")
    if profile.get("cp32_policy") and profile["cp16_policy"] == "carry8":
        raise ValueError("CP32 hypotheses require a paired CP16 GQA response")
    # These are deliberate transfers within the current experiment contract,
    # not universal efficiency factors for arbitrary toy models or topologies.
    shape = (model.hidden, model.q_heads, model.kv_heads, model.head_dim,
             model.gdn_heads, model.gdn_dim, model.ranks_per_node)
    if shape != (4096, 32, 8, 128, 16, 128, 8):
        raise ValueError("CP profiling transfer requires H4096/Q32/KV8/D128/GDN16x128 and 8 GPUs per node")
    fits = profile.get("fits", {})
    if not fits or set(fits) - {"GQA", "GDN"}:
        raise ValueError("No profiling evidence for requested operator")
    for fit in fits.values():
        for cp in ("2", "4", "8"):
            p = fit["parameters"][cp]
            for key in ("quadratic_fraction", "token_fraction", "fixed_ms"):
                if not math.isfinite(p[key]) or p[key] < 0:
                    raise ValueError("Invalid profiling response coefficient")
    if profile["cp16_policy"] != "carry8":
        pairs = profile.get("paired_anchors", [])
        if "GQA" not in fits or len(pairs) != 2 or sorted(p["length"] for p in pairs) != [65536, 262144]:
            raise ValueError("Paired CP16 transfer requires the two GQA anchors")
        if any(not math.isfinite(p["ratio"]) or p["ratio"] <= 0 for p in pairs):
            raise ValueError("Invalid paired CP ratio")


class Response:
    def __init__(self, model):
        self.model = model
        self.profile = model.cp_response
        self.anchor_fit = None
        if self.profile is None or self.profile.get("schema") == H100_SCHEMA:
            return
        if self.profile["cp16_policy"] != "carry8":
            p = self.profile["fits"]["GQA"]["parameters"]["8"]
            h, rate = model.hidden, model.compute_tflops * 1e9
            a = 6 * (h * (model.q_heads + 2*model.kv_heads) * model.head_dim + h*h) / rate
            b = 6 * model.q_heads * model.head_dim / rate
            power = 1 if self.profile["cp16_policy"] == "paired_linear" else 2
            points = []
            for pair in sorted(self.profile["paired_anchors"], key=lambda r: r["length"]):
                n = pair["length"]
                t8 = ((1+p["token_fraction"])*a*n + (1+p["quadratic_fraction"])*b*n*n)/8 + p["fixed_ms"]
                target, ideal = t8*pair["ratio"], (a*n+b*n*n)/16
                points.append(dict(length=n, coordinate=n**power/16, reference_cp8_ms=t8,
                                   target_ms=target, ideal_ms=ideal, residual_ms=target-ideal,
                                   observed_ratio=pair["ratio"]))
            x, y = points
            slope = (y["residual_ms"]-x["residual_ms"])/(y["coordinate"]-x["coordinate"])
            fixed = x["residual_ms"]-slope*x["coordinate"]
            if min(slope, fixed) < 0:
                raise ValueError("Paired response transfer produced a negative residual")
            self.anchor_fit = dict(power=power, slope=slope, fixed_ms=fixed, anchors=points)

    def cost(self, symbol, cp, linear_ms, quadratic_ms, communication_ms, local_communication_ms,
             tokens, squares, cross_node=None):
        ideal = linear_ms + quadratic_ms
        kind = {"G": "GDN", "*": "GQA"}.get(symbol)
        if self.profile is not None and self.profile.get("schema") == H100_SCHEMA:
            if kind is None:
                return ideal + communication_ms
            anchor = self.profile["anchors"][kind]["milliseconds"]
            if cp not in (1, 2, 4) and str(cp) not in anchor:
                raise ValueError(f"H100 CP>{max(map(int, anchor))} requires new evidence or an explicit new hypothesis")
            if cp <= 8 and cross_node:
                raise ValueError("H100 CP<=8 anchor/extrapolation requires node-local groups")
            milliseconds = anchor[str(cp)] if str(cp) in anchor else anchor["8"] * 8 / cp
            n = self.profile["anchor_tokens"]
            # Each anchor already contains the full module and its CP communication.
            # Multiplying by useful-work ratio is a declared shape extrapolation,
            # not a second fitted observation; no analytic communication is added.
            return milliseconds * operator_work(self.model, symbol, tokens, squares) / operator_work(self.model, symbol, n, n*n)
        if self.profile is None or cp == 1 or kind not in self.profile["fits"]:
            return ideal + communication_ms
        if cp == 32:
            policy = self.profile.get("cp32_policy")
            if policy is None:
                raise ValueError("CP32 requires an explicit unmeasured response hypothesis")
            if kind == "GQA":
                f = self.anchor_fit
                coordinate = (tokens if f["power"] == 1 else squares)/16
                variable = f["slope"]*coordinate
                if policy == "work_scaling":
                    residual = variable/2 + f["fixed_ms"]
                elif policy == "carry_residual":
                    residual = variable + f["fixed_ms"]
                else:
                    residual = 2*(variable + f["fixed_ms"])
                # This replaces the WHOLE GQA response, including unknown link
                # and execution effects; do not add modeled communication again.
                return ideal + residual
        elif cp not in (2, 4, 8, 16):
            raise ValueError("CP response transfer supports CP<=32 only")
        if cp == 16 and kind == "GQA" and self.anchor_fit is not None:
            f = self.anchor_fit
            coordinate = (tokens if f["power"] == 1 else squares)/16
            return ideal + f["slope"]*coordinate + f["fixed_ms"]
        p = self.profile["fits"][kind]["parameters"][str(min(cp, 8))]
        residual = p["token_fraction"]*linear_ms + p["quadratic_fraction"]*quadratic_ms + p["fixed_ms"]
        result = ideal + residual + max(0., communication_ms-local_communication_ms)
        return result*self.profile.get("gdn32_multiplier", 1.) if cp == 32 else result


def response_coverage(plan, lengths, model):
    """Counts are group calls across layers, not independent measured samples."""
    if model.cp_response is None:
        return dict(policy="analytic_unvalidated")
    if model.cp_response.get("schema") == H100_SCHEMA:
        counts = dict(anchor_domain_group_calls=0, shape_extrapolated_group_calls=0,
                      cp_extrapolated_group_calls=0, packed_group_calls=0,
                      cp16_gqa_calls=0, cp16_gdn_calls=0)
        domain = tuple(map(int, model.cp_response["anchors"]["GQA"]["milliseconds"]))
        if 32 in domain:
            counts.update(cp32_gqa_calls=0, cp32_gdn_calls=0)
        n = model.cp_response["anchor_tokens"]
        for batch in plan.microbatches:
            for symbol, (layout, _) in zip(plan.pattern, batch.layers):
                for group in layout.groups:
                    cp = len(group.ranks)
                    at_shape = len(group.samples) == 1 and lengths[group.samples[0]] == n
                    counts["anchor_domain_group_calls"] += at_shape and cp in domain
                    counts["shape_extrapolated_group_calls"] += not at_shape
                    counts["cp_extrapolated_group_calls"] += cp not in domain
                    counts["packed_group_calls"] += len(group.samples) > 1
                    counts["cp16_gdn_calls" if symbol == "G" else "cp16_gqa_calls"] += cp == 16
                    if cp == 32:
                        counts["cp32_gdn_calls" if symbol == "G" else "cp32_gqa_calls"] += 1
        return dict(policy=model.cp_response["name"], hardware="H100", counts=counts,
                    source=model.cp_response["source"], anchors=model.cp_response["anchors"],
                    cp32_source=model.cp_response.get("cp32_source"),
                    shape_policy=model.cp_response["shape_policy"],
                    unmeasured_cp_policy=model.cp_response["unmeasured_cp_policy"],
                    other_terms="MLP, head, routing, update and memory are uncalibrated analytic estimates",
                    gpu_measured=False)
    counts = dict(calibrated_group_calls=0, packed_group_calls=0, outside_length_domain_calls=0,
                  cp16_gqa_calls=0, cp16_gdn_calls=0, cp32_gqa_calls=0, cp32_gdn_calls=0,
                  uncalibrated_gdn_calls=0)
    for batch in plan.microbatches:
        for symbol, (layout, _) in zip(plan.pattern, batch.layers):
            kind = "GDN" if symbol == "G" else "GQA"
            for group in layout.groups:
                cp = len(group.ranks)
                if cp == 1:
                    continue
                if kind not in model.cp_response["fits"]:
                    counts["uncalibrated_gdn_calls"] += 1
                    continue
                counts["calibrated_group_calls"] += 1
                counts["packed_group_calls"] += len(group.samples) > 1
                lo, hi = model.cp_response["fits"][kind]["sequence_domain_ki"]
                if cp == 16:
                    counts["cp16_gdn_calls" if kind == "GDN" else "cp16_gqa_calls"] += 1
                    if kind == "GQA" and model.cp_response["cp16_policy"] != "carry8":
                        lo, hi = 64, 256
                elif cp == 32:
                    counts["cp32_gdn_calls" if kind == "GDN" else "cp32_gqa_calls"] += 1
                    if kind == "GQA":
                        lo, hi = 64, 256
                counts["outside_length_domain_calls"] += any(not lo*1024 <= lengths[s] <= hi*1024 for s in group.samples)
    return dict(policy=model.cp_response["name"], counts=counts,
                gdn_head_transfer=model.cp_response["gdn_transfer"],
                anchor_fit=Response(model).anchor_fit, cp32_policy=model.cp_response.get("cp32_policy"),
                cp32_scope=model.cp_response.get("cp32_scope"), gpu_measured=False)
