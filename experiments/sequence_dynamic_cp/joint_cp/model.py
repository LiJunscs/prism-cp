"""Explicit provisional cost and whole-F/B storage model, with no token cap.

These are analytic engineering estimates, NOT hardware measurements. Predicted
feasibility is not GPU admission. Profile assumptions travel with every result.
All times are milliseconds and represent forward + backward unless noted.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from .ir import identity_route, route
from .response import Response, validate_response


@dataclass(frozen=True)
class Model:
    hidden: int = 4096
    ffn: int = 16384
    vocab: int = 32768
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    gdn_heads: int = 16
    gdn_dim: int = 128
    device_bytes: int = 80 * 1024**3
    external_bytes: int = 2 * 1024**3
    margin_bytes: int = 2 * 1024**3
    # Optional observed full model-state allocation (params, grads, Adam, buffers).
    static_bytes: int | None = None
    compute_tflops: float = 300.0
    intra_gbps: float = 300.0
    inter_gbps: float = 25.0
    memory_gbps: float = 1500.0
    ranks_per_node: int = 8
    latency_ms: float = 0.01
    microbatch_ms: float = 0.1
    update_ms: float = 1.0
    provenance: str = "analytic_unvalidated; throughput, link and storage envelopes are assumptions"
    cp_response: dict | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid model field {key}")
        for key in ("hidden", "ffn", "vocab", "q_heads", "kv_heads", "head_dim", "gdn_heads",
                    "gdn_dim", "device_bytes", "compute_tflops", "intra_gbps", "inter_gbps", "memory_gbps", "ranks_per_node"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Model field must be positive: {key}")
        if self.cp_response is not None:
            validate_response(self.cp_response, self)

    @property
    def budget(self):
        return min(int(0.9 * self.device_bytes), self.device_bytes - 8 * 1024**3)

    def static(self, pattern, world):
        if self.static_bytes is not None:
            return self.static_bytes
        h, d = self.hidden, self.gdn_heads * self.gdn_dim
        gqa = h * (self.q_heads + 2 * self.kv_heads) * self.head_dim + h * h
        gdn = h * 5 * d + d * h + 4 * 3 * d + 2 * self.gdn_heads
        params = (2 * self.vocab * h + len(pattern) * (3 * h * self.ffn + 4 * h)
                  + pattern.count("*") * gqa + pattern.count("G") * gdn + h)
        # BF16 weights, FP32 grads, distributed FP32 master/m/v. Small buffer reserve.
        return math.ceil(params * (6 + 12 / world)) + 256 * 1024**2

    def token_capacity_upper_bound(self, pattern, world):
        """Necessary ceiling from average live storage at the output head.

        This relaxes placement imbalance and all other workspaces. It is NOT a
        safe admission capacity; a batch below this ceiling can still fail.
        """
        h, d = self.hidden, self.gdn_heads*self.gdn_dim
        saved = (4*h + pattern.count("G")*2*(8*h+5*d)
                 + pattern.count("*")*(2*(8*h+(self.q_heads+2*self.kv_heads)*self.head_dim)+4*self.q_heads)
                 + len(pattern)*2*(4*h+3*self.ffn))
        head = 4*self.vocab+4*h
        room = self.budget-self.static(pattern, world)-self.external_bytes-self.margin_bytes
        return max(0, int(room*world//(saved+head)))


class Evaluator:
    def __init__(self, model, lengths, world, pattern):
        self.model, self.lengths, self.world, self.pattern = model, lengths, world, pattern
        self.base = model.static(pattern, world) + model.external_bytes + model.margin_bytes
        self.response = Response(model)

    @lru_cache(maxsize=4096)
    def tokens(self, layout):
        return layout.tokens(self.lengths, self.world)

    @lru_cache(maxsize=8192)
    def transfer(self, source, target):
        m = self.model
        zeros = (0,) * self.world
        if source == target:
            return 0.0, zeros
        edges = route(source, target, self.lengths, self.world)
        if identity_route(edges):
            return 0.0, zeros
        row = 2 * m.hidden
        intra, inter = [0] * self.world, [0] * self.world
        # Critical injection/receive time, not sum of traffic across GPUs.
        incoming_intra, incoming_inter = [0] * self.world, [0] * self.world
        for src, dst, _, _, count in edges:
            if src != dst:
                cross = src // m.ranks_per_node != dst // m.ranks_per_node
                (inter if cross else intra)[src] += count * row
                (incoming_inter if cross else incoming_intra)[dst] += count * row
        ms = max((max(a, b) / (m.intra_gbps * 1e6) + max(c, d) / (m.inter_gbps * 1e6)
                  for a, b, c, d in zip(intra, incoming_intra, inter, incoming_inter)), default=0)
        # Local permutations also allocate/copy. Includes forward and inverse backward.
        storage = tuple(2 * row * (a + b) for a, b in zip(self.tokens(source), self.tokens(target)))
        copy_ms = max(storage) / (m.memory_gbps * 1e6)
        return 2 * (ms + copy_ms + m.latency_ms), storage

    @lru_cache(maxsize=4096)
    def operator(self, symbol, layout):
        m, times = self.model, []
        saved, workspace = [0] * self.world, [0] * self.world
        for group in layout.groups:
            cp = len(group.ranks)
            lengths = [self.lengths[s] for s in group.samples]
            n = sum(lengths) // cp
            squares = sum(s*s for s in lengths)
            h, f, d = m.hidden, m.ffn, m.gdn_heads * m.gdn_dim
            quadratic = 0
            if symbol == "M":
                flops, communication = 18 * n * h * f, 0
                save, work = n * 2 * (4 * h + 3 * f), n * 2 * (2 * h + 2 * f)
            else:
                if symbol == "*":
                    projection = 6 * n * (h * (m.q_heads + 2 * m.kv_heads) * m.head_dim + h * h)
                    quadratic = 6 * m.q_heads * m.head_dim * squares / cp
                    flops = projection + quadratic
                    communication = 3 * (cp - 1) * n * 4 * m.kv_heads * m.head_dim
                    save = n * (2 * (8 * h + (m.q_heads + 2 * m.kv_heads) * m.head_dim) + 4 * m.q_heads)
                    work = n * (4 * h + 8 * m.kv_heads * m.head_dim * max(1, cp - 1))
                else:
                    flops = 6 * n * (6 * h * d + 4 * m.gdn_heads * m.gdn_dim**2)
                    communication = 2 * (cp - 1) * len(lengths) * (4 * m.gdn_heads * m.gdn_dim**2 + 6 * 3 * d)
                    save = n * 2 * (8 * h + 5 * d)
                    work = n * 4 * (h + 5 * d) + len(lengths) * cp * 4 * m.gdn_heads * m.gdn_dim**2
            cross = group.ranks[0] // m.ranks_per_node != group.ranks[-1] // m.ranks_per_node
            bandwidth = m.inter_gbps if cross else m.intra_gbps
            latency = 2 * m.latency_ms * (cp - 1) if symbol != "M" else 0
            rate = m.compute_tflops * 1e9
            ms = self.response.cost(symbol, cp, (flops-quadratic)/rate, quadratic/rate,
                                    communication/(bandwidth*1e6)+latency,
                                    communication/(m.intra_gbps*1e6)+latency,
                                    sum(lengths), squares)
            times.append(ms)
            for rank in group.ranks:
                saved[rank], workspace[rank] = save, work
        return max(times), tuple(saved), tuple(workspace)

    def initial(self, ingress):
        m = self.model
        # Embedding/output residual retained throughout the transformer.
        live = tuple(self.base + n * 4 * m.hidden for n in self.tokens(ingress))
        return live

    def layer(self, live, previous, attention, mlp, symbol):
        """Forward prefix plus matching backward point after deeper layers free."""
        a_time, a_save, a_work = self.operator(symbol, attention)
        m_time, m_save, m_work = self.operator("M", mlp)
        incoming, incoming_work = self.transfer(previous, attention)
        outgoing, outgoing_work = self.transfer(attention, mlp)
        after_a = tuple(a + b for a, b in zip(live, a_save))
        after_m = tuple(a + b for a, b in zip(after_a, m_save))
        # MLP returns to the attention residual placement, F and B each have two routes.
        points = [tuple(a + b for a, b in zip(live, incoming_work)),
                  tuple(a + b for a, b in zip(after_a, a_work)),
                  tuple(a + b for a, b in zip(after_a, outgoing_work)),
                  tuple(a + b for a, b in zip(after_m, m_work)),
                  tuple(a + b for a, b in zip(after_m, outgoing_work))]
        peak = tuple(max(values) for values in zip(*points))
        return incoming + a_time + m_time + 2 * outgoing, after_m, peak

    def finish(self, live, previous, ingress):
        transfer, work = self.transfer(previous, ingress)
        m, tokens = self.model, self.tokens(ingress)
        peak = tuple(a + max(b, n * (4 * m.vocab + 4 * m.hidden))
                     for a, b, n in zip(live, work, tokens))
        # Untied output GEMM F+B, plus per-MB overhead; optimizer/sync once per update.
        time = transfer + max(tokens) * 6 * m.hidden * m.vocab / (m.compute_tflops * 1e9) + m.microbatch_ms
        return time, peak

    def evaluate(self, batch):
        live, previous = self.initial(batch.ingress), batch.ingress
        peak, total, points = live, 0.0, []
        for index, (symbol, (attention, mlp)) in enumerate(zip(self.pattern, batch.layers)):
            cost, live, here = self.layer(live, previous, attention, mlp, symbol)
            peak = tuple(max(a, b) for a, b in zip(peak, here))
            total += cost
            points.append(dict(layer=index, saved_live_bytes=list(live), peak_bytes=list(here),
                               attention_tokens=self.tokens(attention), mlp_tokens=self.tokens(mlp)))
            previous = attention
        cost, here = self.finish(live, previous, batch.ingress)
        peak = tuple(max(a, b) for a, b in zip(peak, here))
        return dict(predicted_ms=total + cost, peak_bytes=peak, fits=max(peak) <= self.model.budget,
                    points=points, head_peak_bytes=here, includes_margin=True,
                    memory_status="structural_forward_backward_envelope_not_measured")


def execution_contract(model):
    """Fields the Megatron block must match before replay; full inventory is retained."""
    return dict(hidden_size=model.hidden, ffn_hidden_size=model.ffn,
                num_attention_heads=model.q_heads, num_query_groups=model.kv_heads,
                kv_channels=model.head_dim, linear_num_key_heads=model.gdn_heads,
                linear_num_value_heads=model.gdn_heads, linear_key_head_dim=model.gdn_dim,
                linear_value_head_dim=model.gdn_dim, linear_conv_kernel_dim=4)
