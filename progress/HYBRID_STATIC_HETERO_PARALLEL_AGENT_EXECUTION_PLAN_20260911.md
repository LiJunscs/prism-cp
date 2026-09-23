# Parallel Agent Execution Plan — Hybrid Static Heterogeneous Parallelism

**Date:** 2026-09-11  
**Purpose:** rapidly validate whether a static heterogeneous hybrid-model execution plan can beat the best uniform Megatron configuration before investing in a full runtime  
**Target:** independent H100 workers / nodes; single-node 8×H100 lanes first  
**Coordination rule:** agents do not change shared benchmark semantics without recording a new experiment family

---

## 0. Global operating rules

### 0.1 One coordinator, many independent lanes

Assign one **Coordinator Agent (C0)** with no GPU requirement.

GPU agents work in isolated branches/worktrees and write artifacts only to:

```text
agent_run/<UTC>_<gitsha>_<experiment_id>/
```

No agent overwrites another agent’s results.

### 0.2 Frozen baseline identity

Before official benchmark data is accepted, C0 publishes:

- `environment_manifest.json`
- `model_manifest.json`
- `hardware_manifest.json`
- `benchmark_protocol.md`

Every agent copies the manifest into its run directory.

### 0.3 Latest-Megatron rule

Do not invent or assume old CLI flags.

Each implementation agent must first record:

```bash
git rev-parse HEAD
git status --short
python -c "import torch; print(torch.__version__)"
python -c "import transformer_engine; print(transformer_engine.__version__)"
nvidia-smi
nvidia-smi topo -m
```

Then inspect the exact commit for:

- current GDN/GQA entry points;
- `linear_cp_mode` or successor configuration;
- current CP APIs;
- `HyperCommGrid`;
- process-group collection APIs;
- current FSDP `device_mesh` / sharding API.

If a flag/API differs from this plan, the agent updates `manifest.json`; it does not silently emulate an old API.

### 0.4 Formal timing

Official short benchmark:

- initialization/JIT excluded;
- >=3 warmup iterations;
- >=5 measured iterations;
- median reported.

Final competitive point:

- >=3 independent launches;
- 5 launches if gain is close/noisy.

Profiler runs are diagnostic only and never used as final throughput measurements.

### 0.5 Correctness gate

Every new mechanism must pass, at small shape first:

- output;
- loss if present;
- input gradient;
- parameter gradient;
- one optimizer step where applicable.

A correctness failure blocks performance promotion.

---

# 1. Parallel execution DAG

```text
                    ┌──────── A0: API/environment audit ────────┐
                    │                                           │
                    ├──────── A1: H100 collective calibration ──┤
                    │                                           │
Start ──────────────┼──────── A2: GQA TP×CP frontier ───────────┤
                    │                                           ├─> G0 decision
                    ├──────── A3: GDN TP×CP frontier ───────────┤
                    │                                           │
                    └──────── A4: DTensor transition proto ─────┘
                                                                │
                                          ┌─────────────────────┘
                                          v
                              A5: 3GDN+1GQA manual oracle
                                          │
                                       G1 decision
                                          │
                           ┌───────────────┴───────────────┐
                           v                               v
                 A6: Megatron runtime             A7: correctness suite
                           │                               │
                           └───────────────┬───────────────┘
                                           v
                                  A8: full 14-layer
                                           │
                                       G2 decision
                      ┌────────────────────┼─────────────────────┐
                      v                    v                     v
             A9: FSDP state plane   A10: CSA/HCA       A11: multi-node
                      \                    |                     /
                       \___________________|____________________/
                                           v
                                A12: offline explorer
```

The key rule is that **A9–A12 do not block the earliest paper decision**.

---

# 2. Minimum recommended GPU allocation

If enough H100 nodes are available, start with:

| Agent | GPU allocation | Lane |
|---|---:|---|
| A1 | 8×H100 | collectives |
| A2 | 8×H100 | GQA frontier |
| A3 | 8×H100 | GDN frontier |
| A4 | 8×H100 | DTensor transition |
| A5-prep | CPU initially | macro harness preparation |
| A0 | CPU | API audit |

This gives the fastest D1/D2 decision.

If only two 8-GPU nodes are available:

- Node 1: A2 → A5
- Node 2: A3 → A4
- run A1 opportunistically before/after the main lanes.

If only one 8-GPU node is available:

priority order is:

1. A3 GDN frontier;
2. A2 GQA frontier;
3. A4 transition;
4. A5 macro oracle;
5. A1 full collective sweep.

---

# 3. Shared experiment configuration

C0 defines a machine-readable config:

```yaml
experiment_family: hybrid_static_mesh_v1

hardware:
  world_size: 8
  expected_gpu: H100
  topology: single_nvlink_island

training:
  dtype: bf16
  micro_batch_size: 1
  data_parallel_size: 1
  pipeline_parallel_size: 1

model:
  hidden_size: 4096
  ffn_hidden_size: 16384
  sequence_lengths: [32768, 65536, 131072]
  mixer_pattern: [GDN, GDN, GDN, GQA]

candidate_views:
  - {tp: 1, cp: 8}
  - {tp: 2, cp: 4}
  - {tp: 4, cp: 2}
  - {tp: 8, cp: 1}
```

The actual head counts/state dimensions must come from the frozen real model manifest. A candidate is `ILLEGAL_DIVISIBILITY` rather than modified if head/hidden dimensions are incompatible.

---

# 4. Agent A0 — Latest Megatron API and integration audit

**GPU:** none  
**Can run immediately:** yes  
**Blocks:** A6 implementation choices

## Objective

Produce the exact integration map for the pinned latest Megatron commit.

## Tasks

1. Locate and summarize:
   - `HyperCommGrid`;
   - `register_view`;
   - process-group collection;
   - HybridModel layer construction;
   - GDN current chunkwise CP code path;
   - GQA current CP code path;
   - FSDP `device_mesh`, `dp_shard_dim`, `dp_outer_dim`;
   - checkpoint/state-dict ownership hooks.

2. Answer:
   - Can multiple TP/CP views over the same 8 ranks be pre-created?
   - Which modules still read global `parallel_state` and therefore assume one global TP world size?
   - Which TP/CP collectives accept explicit process-group handles?
   - What minimum code patch is required to bind a layer group to a non-default view?
   - Can per-module FSDP DeviceMesh be expressed in the current public API?

3. Do not implement a planner.

## Deliverables

```text
audit/megatron_api_map.md
audit/global_state_assumptions.md
audit/minimal_patch_plan.md
audit/api_symbols.json
```

## Pass condition

A6 has a concrete implementation option ranked:

1. no Megatron patch;
2. narrow process-group injection patch;
3. custom prototype wrapper.

---

# 5. Agent A1 — H100 communication calibration

**GPU:** 8×H100  
**Dependency:** none

## Objective

Build the empirical primitive table used by every later cost estimate.

## Matrix

Operations:

- all-gather;
- reduce-scatter;
- all-reduce;
- all-to-all;
- send/recv if transition implementation needs it.

Group sizes:

- 2;
- 4;
- 8.

Payload:

- 4 MiB;
- 16 MiB;
- 64 MiB;
- 256 MiB;
- 1 GiB.

## Protocol

For every point:

- warmup >=20;
- measured >=100 operations if operation duration permits;
- synchronize only at measurement boundaries;
- report p50/p95.

## Deliverables

```text
A1_collectives/collective_table.csv
A1_collectives/topology.txt
A1_collectives/summary.md
```

Required CSV columns:

```text
op,group_size,payload_bytes,p50_us,p95_us,alg_bw_GBs,bus_bw_GBs,peak_workspace_bytes
```

## Pass condition

No missing group size required by A4/A5.

---

# 6. Agent A2 — GQA TP×CP frontier

**GPU:** 8×H100  
**Dependency:** frozen manifest only

## Objective

Find the best legal intra-sample factorization for the current GQA layer/block.

## Candidate matrix

For each target sequence length permitted by budget:

```text
TP1 CP8
TP2 CP4
TP4 CP2
TP8 CP1
```

Start at 32K or 64K. Run 128K only after the harness is validated.

## Required controls

Fixed:

- DP=1;
- PP=1;
- MBS=1;
- dtype;
- checkpoint policy;
- same exact inputs;
- same layer dimensions;
- same kernel/backend where possible.

Do not change batch semantics to make a point fit.

## Typical Megatron semantic knobs

Verify exact names at the pinned commit before use:

- tensor model parallel size;
- context parallel size;
- micro batch size;
- sequence length;
- BF16;
- attention/CP backend.

## Metrics

- fwd ms;
- bwd ms;
- fwd+bwd ms;
- max-rank peak allocated;
- max-rank peak reserved;
- collective exposed wait if measurable;
- correctness vs single/reference configuration.

## Deliverables

```text
A2_gqa/gqa_frontier.csv
A2_gqa/gqa_correctness.json
A2_gqa/gqa_summary.md
```

## Pass condition

At least two legal `(TP,CP)` points measured; all promoted points correct.

---

# 7. Agent A3 — GDN TP×CP frontier

**GPU:** 8×H100  
**Dependency:** frozen manifest only  
**Priority:** highest single performance lane

## Objective

Measure the current **chunkwise** GDN path under the same candidate TP×CP factorizations.

## Candidate matrix

```text
TP1 CP8
TP2 CP4
TP4 CP2
TP8 CP1
```

subject to real model divisibility.

## Mandatory restriction

Do not reuse historical headwise GDN performance as current evidence.

Use the latest production GDN path/configuration at the pinned Megatron commit.

## Extra instrumentation

Separate, if possible:

- GDN core compute;
- recurrent-state/context communication;
- TP collectives;
- recomputation;
- workspace memory.

## Deliverables

```text
A3_gdn/gdn_frontier.csv
A3_gdn/gdn_correctness.json
A3_gdn/gdn_summary.md
```

## Early signal

C0 computes:

```text
best_GDN_view
best_GQA_view
gap_to_shared_view
```

If best views differ by less than benchmark noise and a common view is within ~2%, heterogeneous ActivationMesh is immediately weak.

---

# 8. Agent A4 — DTensor / explicit-collective transition prototype

**GPU:** 8×H100  
**Dependency:** none for implementation; A1 improves interpretation

## Objective

Measure the minimum realistic cost of changing activation ownership between predeclared TP/CP views.

## Required transitions

Priority:

1. `(TP1,CP8) ↔ (TP4,CP2)`
2. `(TP1,CP8) ↔ (TP2,CP4)`
3. `(TP2,CP4) ↔ (TP4,CP2)`

## Implementation options

### Option 1 — DTensor

Use `DeviceMesh` and explicit placements/redistribution where the desired mapping is expressible.

### Option 2 — explicit `torch.distributed`

Use all-to-all / all-gather / reduce-scatter / reshape as required.

### Option 3 — Megatron primitives

Only if they can be used without first building A6.

## Shapes

Primary hidden activation:

```text
[B=1, S, H=4096]
```

BF16.

Run at:

- 32K;
- 64K;
- 128K if memory permits.

## Correctness

Test:

```text
x -> transition A→B -> transition B→A -> x'
```

and backward through the transition.

Compare:

- round-trip output;
- input gradient.

## Metrics

- A→B fwd;
- B→A fwd;
- backward inverse;
- round-trip;
- bytes moved;
- peak workspace;
- temporary allocation count if available.

## Deliverables

```text
A4_transition/transition_table.csv
A4_transition/transition_correctness.json
A4_transition/implementation_notes.md
```

## Pass condition

At least the transition between the observed A2/A3 best views is correct and measured.

---

# 9. Gate G0 — earliest opportunity decision

**Owner:** C0  
**Inputs:** A2, A3, A4; A1 optional for explanation

Compute:

\[
T_{\rm uniform}(s)=3T_{\rm GDN}(s)+T_{\rm GQA}(s)
\]

and first-order heterogeneous estimate:

\[
T_{\rm mixed}
=
3T_{\rm GDN}(s_D)
+T_{\rm GQA}(s_A)
+T_{A\rightarrow D}
+T_{D\rightarrow A}
\]

Do this with measured values, not theoretical bandwidth.

## G0 outcomes

### GREEN

Predicted mixed gain >=8%.

Immediately prioritize A5.

### YELLOW

Predicted gain 3–8%.

A5 still runs, but runtime work waits for real macro result.

### RED

Predicted gain <3%, same optimal view, or transition dominates.

Stop A6. Continue A9 or A10 as alternative research directions.

Deliver:

```text
decisions/G0_opportunity.md
```

---

# 10. Agent A5 — Manual 3GDN+1GQA heterogeneous oracle

**GPU:** 8×H100  
**Dependency:** A2/A3/A4  
**This is the decisive experiment.**

## Objective

Test whether heterogeneous execution wins when real layer interactions and transitions are present.

## Workload

```text
GDN
GDN
GDN
GQA
```

Use production-faithful hidden/FFN dimensions.

First run a block-only version; then include FFN/residual/norm if not already included.

## Cases

### U-best

Best measured uniform `(TP,CP)` over the whole macro-block.

### U-alt

Second-best uniform plan.

### H-manual

Best manual heterogeneous plan based on A2/A3.

### H-greedy

Per-mixer local best, if this differs from H-manual/global-aware grouping.

## Implementation

Hard-wired prototype is acceptable.

Do **not** wait for generic configuration APIs.

## Metrics

Same as official protocol plus:

- transition fraction of macro time;
- peak memory during transition;
- full backward.

## Final repetitions

If H-manual appears >5% faster:

- run 3 independent launches;
- alternate U/H launch ordering.

## Deliverables

```text
A5_oracle/macro_results.csv
A5_oracle/macro_correctness.json
A5_oracle/gain_breakdown.md
```

---

# 11. Gate G1 — build runtime or stop

**Owner:** C0

### GREEN

H-manual:

- correctness PASS;
- stable >=5% faster than U-best;
- no unacceptable memory regression.

Start A6 + A7 immediately.

### STRONG GREEN

>=8–10% stable gain.

Promote heterogeneous Activation Plane as primary paper mechanism.

### RED

<5% or unstable.

Do not build general TP↔CP runtime yet.

Redirect GPU budget to A9/A10.

Deliver:

```text
decisions/G1_manual_oracle.md
```

---

# 12. Agent A6 — Megatron static heterogeneous runtime

**GPU:** development can begin CPU-side; 8×H100 for validation  
**Dependency:** G1 GREEN, A0 audit

## Objective

Replace A5 hard wiring with a reusable predeclared execution-plan mechanism.

## Minimum scope

Only implement views actually needed by A5.

Example:

```yaml
views:
  gqa:
    tp: 1
    cp: 8
  gdn:
    tp: 4
    cp: 2

bindings:
  - layers: [0,1,2]
    view: gdn
  - layers: [3]
    view: gqa
```

## Preferred integration order

1. use `HyperCommGrid.register_view()` where possible;
2. inject explicit PG collections into target modules;
3. patch global TP/CP assumptions only where necessary;
4. do not redesign unrelated Megatron code.

## Hard requirements

- all views registered at startup;
- all process groups created at startup;
- no planner code in forward;
- no runtime process-group create/destroy;
- transition buffers preallocated where practical;
- fixed collective order.

## Performance target

Generic runtime plan <=2% slower than A5 manual hard-wired oracle.

## Deliverables

```text
A6_runtime/parallel_plan_schema.md
A6_runtime/patch.diff
A6_runtime/runtime_overhead.csv
A6_runtime/integration_notes.md
```

---

# 13. Agent A7 — Correctness certification

**GPU:** 8×H100, but many small tests are cheap  
**Dependency:** A6 prototype

## Objective

Attempt to break the heterogeneous runtime semantically.

## Matrix

At minimum:

- small deterministic shape;
- BF16 production-like shape;
- backward;
- one optimizer step;
- multiple gradient-accumulation microbatches;
- activation checkpointing;
- save/resume smoke test.

## Compare

Reference uniform or equivalent mathematical execution vs heterogeneous plan:

- output;
- loss;
- input grad;
- param grad;
- post-step weights;
- optimizer state;
- resumed next-step loss.

## Deliverables

```text
A7_correctness/correctness_matrix.csv
A7_correctness/post_step_parity.json
A7_correctness/resume_parity.json
A7_correctness/failures.md
```

## Blocker rule

Any unexplained correctness mismatch blocks paper performance promotion.

---

# 14. Agent A8 — Full 14-layer GDN/GQA training

**GPU:** 8×H100  
**Dependency:** A6 + A7 PASS

## Objective

Prove that the macro-block gain survives the real full-model critical path.

## Sequence order

1. 32K
2. 64K
3. 128K

Do not start at 128K if shorter production-faithful cases fail.

## Compare

- best tuned uniform Megatron;
- manual heterogeneous oracle if separately maintained;
- generic static heterogeneous runtime.

Keep:

- same model init;
- same optimizer state;
- same sample sequence;
- same global tokens;
- same checkpoint/recompute policy.

## Acceptance

Primary performance Go:

- >=5% stable step-time gain vs best uniform Megatron.

Strong claim:

- >=8–10% on a primary setting.

## Deliverables

```text
A8_full_model/full_model_results.csv
A8_full_model/memory_results.csv
A8_full_model/correctness_results.csv
A8_full_model/summary.md
```

---

# 15. Agent A9 — FSDP Param/State Plane

**GPU:** 8×H100  
**Dependency:** can begin after A0; does not block A5/A6

## Objective

Test whether different mixer families prefer different static parameter/state sharding domains while Activation Plane is fixed.

## Phase F1 — fixed activation

Start with:

```text
DP=1
TP=1
CP=8
MBS=1
```

or the best fixed activation layout supported by the real stack.

Candidate state plans:

- FSDP8;
- FSDP4 × state-replica2;
- FSDP2 × state-replica4;
- supported HSDP/HFSDP variants.

## Important semantic point

The replica factor is **model-state replication**, not extra independent samples.

Each CP rank still owns its real context shard.

## Measure separately for

- GQA block;
- GDN block.

## Questions

- best FSDP degree per mixer?
- exposed AG/RS?
- peak state memory?
- can current Megatron-FSDP express per-module mesh?
- does mixed per-layer FSDP improve a macro block?

## Deliverables

```text
A9_fsdp/fsdp_frontier_gqa.csv
A9_fsdp/fsdp_frontier_gdn.csv
A9_fsdp/fsdp_correctness.json
A9_fsdp/per_module_mesh_gap.md
```

## Promotion gate

Only continue to heterogeneous FSDP full model if mixed state plan is expected to gain >=5% or unlock meaningful capacity.

---

# 16. Agent A10 — CSA/HCA secondary workload

**GPU:** 8×H100  
**Dependency:** independent of A6 unless testing heterogeneous hidden mesh

## Objective

Test whether a second hybrid family needs hidden-layout heterogeneity or only representation/backend specialization.

## First experiment

Keep hidden TP/CP fixed.

Profile:

- HCA compressed context;
- CSA compressed context + selection;
- transition-free alternating `CSA-HCA-CSA-HCA`.

## Then

Only if local hidden-layout optima differ strongly:

- measure potential hidden-layout transition;
- compare uniform hidden layout vs mixed layout.

## Valid successful outcomes

1. heterogeneous hidden layout wins;
2. fixed hidden layout wins because transitions are too frequent.

Outcome 2 is useful evidence for the global reconciler.

## Deliverables

```text
A10_csa_hca/operator_frontier.csv
A10_csa_hca/alternating_macro.csv
A10_csa_hca/summary.md
```

MoE/EP must remain disabled/excluded for this phase.

---

# 17. Agent A11 — Multi-node topology validation

**GPU:** 2+ H100 nodes  
**Dependency:** A8 GREEN

## Objective

Check whether the single-node design survives real topology without introducing a new mechanism.

## First configuration

Keep high-frequency TP×CP view inside each 8-GPU NVSwitch island.

Scale using:

- DP and/or PP across nodes.

Do not start with cross-node TP/CP.

## Compare

- uniform Megatron multi-node;
- same heterogeneous plan structure.

## Extra measurements

- inter-node exposed communication;
- node-local vs cross-node traffic;
- scaling efficiency.

## Deliverables

```text
A11_multinode/scaling.csv
A11_multinode/topology_trace.md
A11_multinode/summary.md
```

---

# 18. Agent A12 — Offline explorer

**GPU:** none for search; uses existing measurements  
**Dependency:** A5 at minimum, preferably A8

## Objective

Show that manual plan selection can be automated without entering the runtime.

## Input tables

- local mixer candidate latency;
- memory;
- transition latency/memory;
- legality flags;
- layer sequence.

## Candidate pruning

Per mixer retain:

- Pareto frontier;
- top-k (2–5) within a configurable gap to local best.

## Solver

Simple DP / shortest path:

\[
D[g,s]=T_g(s)+\min_{s'}(D[g-1,s']+R(s',s))
\]

plus memory/legality constraints.

## Validation

On a small candidate set:

- exhaustive enumerate all legal plans;
- compare explorer plan to best measured/exhaustive plan.

## Deliverables

```text
A12_explorer/explorer.py
A12_explorer/plan.yaml
A12_explorer/plan_quality.csv
A12_explorer/README.md
```

Target:

- selected plan within 1–2% of best measured/exhaustive plan.

---

# 19. Coordinator C0 — Continuous aggregation

C0 runs whenever new artifacts land.

## Required summary tables

### `frontier_merged.csv`

```text
mixer,seq,tp,cp,status,fwd_ms,bwd_ms,total_ms,peak_alloc,peak_reserved
```

### `transition_merged.csv`

```text
seq,src_tp,src_cp,dst_tp,dst_cp,status,fwd_ms,bwd_ms,roundtrip_ms,peak_workspace
```

### `macro_merged.csv`

```text
plan_id,seq,status,step_ms,peak_alloc,gain_vs_best_uniform,correctness
```

## Auto-generated decision memo

After every material result update:

```text
decisions/current_status.md
```

with:

- best GQA view;
- best GDN view;
- best shared view;
- predicted mixed gain;
- measured macro gain;
- current Go/No-Go;
- next GPU task.

---

# 20. Suggested experiment IDs

Use stable IDs:

```text
A1_COLL_<op>_G<group>_<size>
A2_GQA_S<seq>_TP<tp>_CP<cp>
A3_GDN_S<seq>_TP<tp>_CP<cp>
A4_TR_S<seq>_<src>_TO_<dst>
A5_MACRO_S<seq>_<plan>
A6_RT_S<seq>_<plan>
A7_CORR_<case>
A8_FULL_S<seq>_<plan>
A9_FSDP_<mixer>_F<f>_R<r>
A10_CSAHCA_<case>
A11_MN_N<nodes>_<plan>
A12_PLAN_<dataset>
```

---

# 21. Fastest possible first 24-result package

The project should aim to collect the following **before any large runtime refactor**:

1. 4 GQA frontier points;
2. 4 GDN frontier points;
3. 3 primary transition pairs × both directions;
4. at least 2 uniform macro plans;
5. 1 manual heterogeneous macro plan;
6. correctness for all promoted points.

This is enough to answer the core paper question.

---

# 22. Stop / redirect policy

### Stop heterogeneous ActivationMesh engineering if

- GQA/GDN best layouts are the same;
- a common layout is within ~2% of both local optima;
- measured macro gain <5%;
- transition makes 128K infeasible;
- correctness requires changing training semantics.

### Redirect to A9 if

- state memory/AG/RS dominates;
- per-mixer FSDP sharding sweet points differ.

### Redirect to A10 if

- hybrid value lies mainly in context representation/backend rather than hidden mesh.

### Do not resurrect TMF unless

- a proven profitable layout requires DP expansion;
- TP/head/state intra-sample parallelism cannot absorb the ranks;
- MBS=1 blocks the spatial conversion;
- measured gain still exceeds temporal and memory cost.

---

# 23. Final paper-ready evidence package

If the project reaches paper stage, freeze:

```text
artifact/
    commits.txt
    environment/
    configs/
    raw_runs/
    merged_tables/
    correctness/
    planner/
    figures/
    reproduction.md
```

Minimum headline comparisons:

1. best uniform Megatron;
2. manual heterogeneous oracle;
3. generic static heterogeneous runtime;
4. per-mixer greedy;
5. offline explorer plan;
6. small exhaustive oracle;
7. FSDP/CSA-HCA secondary evidence if promoted.

The paper must always report speedup against **best tuned uniform Megatron under the same semantics**, never only against a default recipe.
