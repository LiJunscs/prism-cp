"""Frozen native N and joint CP replay with full-update training guards."""

import functools
import json
import os
import time
from pathlib import Path

# Packet mode replaces digest fingerprints before imported bindings are formed.
if os.environ.get("SEQUENCE_CP_PACKET_MANIFEST") or os.environ.get("SEQUENCE_CP_NO_DIGEST") == "1":
    from mixed_global_batch import install_identity
    install_identity()

from planner import Constraints, from_rank_ids, fingerprint, to_rank_ids, validate


def install(config_path: Path) -> None:
    """Install frozen replay, preserving native packing/loss/parameter synchronization."""
    import torch
    from megatron.core.datasets import data_schedule

    settings = json.loads(config_path.read_text())
    if not settings.get("two_dimensional_input") or not settings.get("replay") or not settings.get("mixed_cp"):
        raise ValueError("Runtime requires a frozen native N or joint packet replay")
    retired = ("online_capacity", "arch_capacity", "runtime_capacity_16gpu", "capacity_gpu_first", "input_stream", "packet_manifest", "theory_pattern")
    if any(settings.get(key) for key in retired) or settings.get("input_transport", "native") != "native":
        raise ValueError("Retired experiment mode; use native N or joint replay")
    mode = settings.get("two_dimensional", {}).get("mode")
    if mode not in (None, "joint") or bool(settings.get("native_n")) == (mode == "joint"):
        raise ValueError("Select exactly one of native N and the joint executor")
    if any(settings.get("two_dimensional", {}).get(key) for key in ("cost_diagnostic", "memory_diagnostic", "memory_admission_diagnostic")):
        raise ValueError("Dedicated diagnostic experiments have been retired")
    hbm_guard_enabled = True
    if settings.get("two_dimensional"):
        if settings["two_dimensional"].get("mode") == "joint":
            from joint_cp.adapter import install as install_joint
            from joint_cp.replay import validate_replay

            joint_replays = settings.get("replay", [])
            joint_plans = [validate_replay(row, int(os.environ["WORLD_SIZE"])) for row in joint_replays]
            if not settings.get("two_dimensional_input") or not settings.get("mixed_cp"):
                raise ValueError("Joint replay requires the frozen packet input and mixed kernels")
            joint_position = {"update": None, "ordinal": 0}

            def joint_batch_provider(consume):
                update = execution_state["update"]
                if joint_position["update"] != update:
                    joint_position.update(update=update, ordinal=0)
                ordinal = joint_position["ordinal"]
                plan = joint_plans[update - 1]
                if ordinal >= len(plan.microbatches):
                    raise ValueError("Joint executor consumed more microbatches than its plan")
                if consume:
                    joint_position["ordinal"] += 1
                return plan.microbatches[ordinal], execution_state["lengths"]

            install_joint(settings["two_dimensional"], joint_replays, joint_batch_provider)
        # eval_iters=0 creates empty validation datasets. They must not form a
        # sampler (which rejects zero samples) or consume a training input.
        from megatron.training.datasets import data_samplers
        from megatron.training import training

        original_loader = data_samplers.build_pretraining_data_loader

        def two_dimensional_loader(dataset, consumed_samples):
            if dataset is None or len(dataset) == 0:
                return None
            return original_loader(dataset, consumed_samples)

        data_samplers.build_pretraining_data_loader = two_dimensional_loader
        training.build_pretraining_data_loader = two_dimensional_loader
    if settings.get("native_n"):
        # Empty validation datasets must never create a sampler or consume input.
        from megatron.training.datasets import data_samplers
        from megatron.training import training
        original_loader = data_samplers.build_pretraining_data_loader
        def native_n_loader(dataset, consumed_samples):
            if dataset is None or len(dataset) == 0:
                return None
            return original_loader(dataset, consumed_samples)
        data_samplers.build_pretraining_data_loader = native_n_loader
        training.build_pretraining_data_loader = native_n_loader
    input_transport = "native"
    packet_source = None
    if settings.get("two_dimensional_input"):
        from two_dimensional_packet import install as install_two_dimensional_packets
        packet_source = install_two_dimensional_packets(settings["two_dimensional_input"], settings["packet_updates"],
                                                        int(os.environ["WORLD_SIZE"]))
    original = data_schedule.scheduler_map["default_dynamic_cp"]
    calls = 0
    execution_state = {"packet_warmup": settings.get("packet_warmup",10), "parameter_version": 0,
                       "capacity_preflight": False}
    def diagnostic_update():
        return os.environ.get("SEQUENCE_CP_COMM_AUDIT") == "1" and execution_state.get("update") in (6, 7)

    class SequenceDynamicCPScheduler(original):
        def run(self, *args, **kwargs):
            units = settings.get("accumulation_units", 1)
            # Evaluation always consumes one unit, without changing training data.
            units = units if execution_state.get("training_active", False) else 1
            execution_state["accumulation_units"] = units
            execution_state["unit_index"] = 0
            first = self.run_unit(*args, **kwargs)
            if units == 1:
                if diagnostic_update() and execution_state.get("training_active"):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    execution_state["unit_gpu_events"].append([event, None])
                    torch.cuda.nvtx.mark(f'accum_unit_begin/update={execution_state["update"]}/unit=0')
                return first
            iterator, count, tokens, squares = first
            if isinstance(iterator, list):
                assert len(iterator) == 1
                iterator = iterator[0]
            fingerprint0 = execution_state["fingerprint"]
            execution_state["unit_microbatches"] = count
            execution_state["unit_source_microbatches"] = len(execution_state["plan"])
            def stream():
                for unit in range(units):
                    execution_state["unit_index"] = unit
                    execution_state["unit_start_ns"] = time.perf_counter_ns()
                    if accumulation.optimizer_steps:
                        raise RuntimeError("Parameter update inside accumulation units")
                    if unit == 0:
                        current = iterator
                    else:
                        current, n, t, sq = self.run_unit(*args, **kwargs)
                        assert (n, t, sq) == (count, tokens, squares)
                        assert execution_state["fingerprint"] == fingerprint0
                    if diagnostic_update():
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        execution_state["unit_gpu_events"].append([event, None])
                        torch.cuda.nvtx.mark(f'accum_unit_begin/update={execution_state["update"]}/unit={unit}')
                    for microbatch in range(count):
                        execution_state["unit_microbatch"] = microbatch
                        yield next(current)
                    if diagnostic_update():
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                        execution_state["unit_gpu_events"][-1][1] = end
                        torch.cuda.nvtx.mark(f'accum_unit_end/update={execution_state["update"]}/unit={unit}')
                    execution_state.setdefault("unit_intervals", []).append(dict(
                        unit=unit, parameter_version=execution_state["update"]-1, begin_ns=execution_state["unit_start_ns"], end_ns=time.perf_counter_ns(),
                        logical_sample_ids=list(range(((execution_state["update"]-1)*units+unit)*len(execution_state["lengths"]),
                            ((execution_state["update"]-1)*units+unit+1)*len(execution_state["lengths"])))))
            return type(iterator)(stream()), count * units, tokens * units, squares * units

        def run_unit(self, *args, **kwargs):
            config = kwargs.get("config", args[7] if len(args) > 7 else None)
            if config is None:
                raise ValueError("Missing model config")
            if config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
                raise ValueError("First sequence-CP adapter supports TP=PP=1 only")
            if config.virtual_pipeline_model_parallel_size is not None:
                raise ValueError("VPP alignment is not yet supported by this adapter")
            if getattr(config, "gdn_dynamic_context_parallel", False):
                raise ValueError("Sequence replay requires the legacy GDN child CP path disabled")
            for name, expected in settings.get("model_config", {}).items():
                if getattr(config, name, None) != expected:
                    raise ValueError(f"Model config mismatch for {name}: expected {expected}")
            if packet_source is not None and (not execution_state.get("training_active") or calls != execution_state["update"]-1):
                raise ValueError("Packet scheduler must run once per optimizer update and never for eval")
            result = super().run(*args, **kwargs)
            if not settings.get("mixed_cp", False):
                execution_state.pop("execution_rounds", None)
                return result
            if not config.calculate_per_token_loss:
                raise ValueError("Mixed CP requires per-token loss normalization")
            if (getattr(config, "packed_sequence_cp_size", None) is not None
                or config.recompute_granularity is not None
                or getattr(config, "num_moe_experts", None)
                or getattr(config, "mtp_num_layers", None)):
                raise ValueError("Mixed CP currently supports dense GQA/GDN without child CP, MTP or recompute")
            from mixed_plan import coalesce_rounds

            iterator, count, tokens, squares = result
            if settings.get("two_dimensional", {}).get("mode") == "joint":
                # The joint solver owns microbatch boundaries. Never coalesce them
                # again using an ingress-only capacity or an N packing heuristic.
                rounds = tuple((i,) for i in range(count))
            else:
                rounds = coalesce_rounds(
                    execution_state["plan"], execution_state["lengths"], self.total_hdp_gpus,
                    self.max_seq_len_per_rank, self.max_num_seqs,
                )
            if isinstance(iterator, list):
                if len(iterator) != 1:
                    raise ValueError("Mixed CP does not support multiple model chunks")
                iterator = iterator[0]
            samples = [next(iterator) for _ in range(count)]
            merged = [samples[r[0]] if len(r) == 1 else
                      {"mixed_cp_batches": [samples[i] for i in r]} for r in rounds]
            execution_state["execution_rounds"] = rounds
            if packet_source is not None  and [list(r) for r in rounds] != execution_state["packet_replay"]["execution_rounds"]:
                raise ValueError("Compiled packet execution differs from frozen CPU rounds")
            if int(os.environ.get("RANK", "0")) == 0:
                record = dict(
                    fingerprint=execution_state["fingerprint"], source_microbatches=count,
                    microbatches=len(rounds), execution_rounds=rounds,
                    segments_by_rank=[[
                        [dict(cp=len(g.ranks), ranks=g.ranks, sample_ids=g.ids,
                              local_tokens=sum(execution_state["lengths"][i] for i in g.ids)//len(g.ranks))
                         for index in r for g in execution_state["plan"][index] if rank in g.ranks]
                        for rank in range(self.total_hdp_gpus)] for r in rounds],
                )
                with (Path(os.environ["SEQUENCE_CP_OUTPUT"]) / "mixed_execution.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            return type(iterator)(iter(merged)), len(rounds), tokens, squares

        def get_groups_and_subsamples(self, sample_id_seqlens):
            nonlocal calls
            calls += 1
            started = time.perf_counter()
            lengths = dict(sample_id_seqlens)
            if len(lengths) != len(sample_id_seqlens):
                raise ValueError("Duplicate global sample IDs")
            limits = Constraints(
                self.total_hdp_gpus,
                self.max_seq_len_per_rank,
                self.min_cp_size,
                self.max_num_seqs,
                2 * self.total_hdp_gpus,
            )
            if packet_source is not None:
                # Explicit index, no fixed-count offset or length-based first match.
                step_id = execution_state["update"]
                if not 1 <= step_id <= len(packet_source):
                    raise ValueError("Packet replay exhausted")
                fixed = settings["replay"][step_id-1]
                expected_stream = "TWO_DIMENSIONAL" if settings.get("two_dimensional_input") else "MIX_CORE"
                if (fixed["step"] != step_id or fixed["stream_id"] != expected_stream
                    or fixed["architecture"] != settings["packet_architecture"]
                    or fixed["role"] != settings["packet_role"]):
                    raise ValueError("Packet replay identity mismatch")
                samples = packet_source.samples[step_id-1]
                ids = [s["logical_sample_id"] for s in samples]
                if list(lengths) != list(range(len(samples))) or list(lengths.values()) != fixed["expected_lengths"] or ids != fixed["logical_sample_ids"]:
                    raise ValueError("Packet replay input/order/mapping mismatch")
                plan = from_rank_ids(fixed["rank_ids"])
                if settings.get("two_dimensional", {}).get("mode") == "joint":
                    from joint_cp.replay import validate_replay
                    validate_replay(fixed, self.total_hdp_gpus)
                    self.max_seq_len_per_rank = fixed["capacity_tokens"]
                    limits = Constraints(self.total_hdp_gpus, self.max_seq_len_per_rank,
                                         self.min_cp_size, self.max_num_seqs, 2 * self.total_hdp_gpus)
                validate(plan,lengths,limits)
                if settings.get("two_dimensional_input"):
                    if to_rank_ids(plan, limits.world) != fixed["rank_ids"]:
                        raise ValueError("Two-dimensional packet ownership differs from frozen replay")
                    actual = fixed["candidate_id"]
                else:
                    actual = fingerprint(plan,lengths)
                    if actual != fixed["fingerprint"]:
                        raise ValueError("Packet structural identity mismatch")
                if settings["packet_role"] == "N" and not fixed["native_verified_cpu"]:
                    raise ValueError("N must be verified at chosen generation capacity on CPU")
                execution_state.update(plan=plan,lengths=lengths,fingerprint=actual,
                    logical_sample_ids=ids,packet_replay=fixed,packet_lookup_ms=(time.perf_counter()-started)*1000)
                return to_rank_ids(plan,limits.world)
            raise ValueError("Packet source was not installed")

    data_schedule.scheduler_map["default_dynamic_cp"] = SequenceDynamicCPScheduler

    if settings.get("record_training_steps", False):
        from logical_numerics import LogicalNumerics
        from megatron.training import training
        from numerics import TrainingNumerics

        communication = None
        if os.environ.get("SEQUENCE_CP_COMM_AUDIT") == "1":
            from communication_audit import CommunicationAudit
            communication = CommunicationAudit(Path(os.environ["SEQUENCE_CP_OUTPUT"]), int(os.environ["SEQUENCE_CP_COMM_STEP"]))
        integrity = None
        if packet_source is not None:
            from mixed_packet_integrity import PacketIntegrity
            integrity = PacketIntegrity(execution_state,None)
            execution_state["packet_integrity"] = integrity
        elif settings.get("input_stream") or settings.get("mixed_cp"):
            from stream_integrity import StreamIntegrity
            integrity = StreamIntegrity(execution_state, None)
        from accumulation_audit import AccumulationAudit
        accumulation = AccumulationAudit(execution_state, Path(os.environ["SEQUENCE_CP_OUTPUT"]))
        units = settings.get("accumulation_units", 1)
        if units > 1:
            original_sample_targets = training.get_train_valid_test_num_samples
            def sample_targets():
                train, valid, test = original_sample_targets()
                return train * units, valid, test
            training.get_train_valid_test_num_samples = sample_targets
            original_lr_scheduler = training.get_optimizer_param_scheduler
            def lr_scheduler(optimizer):
                scheduler = original_lr_scheduler(optimizer)
                # Keep the LR/WD schedule in optimizer-update units while exposing
                # the actual effective sample increment for the larger batch.
                for name in ("lr_decay_steps", "lr_warmup_steps", "wd_incr_steps"):
                    setattr(scheduler, name, getattr(scheduler, name) * units)
                original_increment = scheduler.step
                def increment(increment):
                    return original_increment(increment=increment * units)
                scheduler.step = increment
                return scheduler
            training.get_optimizer_param_scheduler = lr_scheduler
        original_step = training.train_step
        step = 0
        numeric = TrainingNumerics(Path(os.environ["SEQUENCE_CP_OUTPUT"]))
        logical = LogicalNumerics(execution_state, Path(os.environ["SEQUENCE_CP_OUTPUT"]))

        @functools.wraps(original_step)
        def train_step(*args, **kwargs):
            nonlocal step
            step += 1
            model_chunks = kwargs.get("model", args[2] if len(args) > 2 else None)
            if step == 1 and (int(os.environ["RANK"]) == 0 or packet_source is not None):
                inventory = []
                for chunk in model_chunks:
                    core = chunk
                    while hasattr(core, "module"):
                        core = core.module
                    inventory.append(
                        dict(
                            model_class=type(core).__name__,
                            attention_classes=[type(layer.self_attention).__name__ for layer in core.decoder.layers],
                            parameter_count=sum(v.numel() for v in core.parameters()),
                            actual_config={k:getattr(core.config,k,None) for k in ("num_layers","hidden_size","ffn_hidden_size","num_attention_heads","num_query_groups","linear_num_key_heads","linear_num_value_heads","linear_key_head_dim","linear_value_head_dim","linear_conv_kernel_dim","recompute_granularity","bf16")},
                            share_embeddings_and_output_weights=getattr(core,'share_embeddings_and_output_weights',None),
                        )
                    )
                if settings.get("expected_attention_classes") and inventory[0]["attention_classes"] != settings["expected_attention_classes"]:
                    raise ValueError("Actual attention layer types differ from explicit pattern")
                folder = Path(os.environ["SEQUENCE_CP_OUTPUT"])
                folder.mkdir(parents=True, exist_ok=True)
                if packet_source is not None:
                    (folder / f'model_inventory_rank{os.environ["RANK"]}.json').write_text(json.dumps(inventory, indent=2)+"\n")
                if int(os.environ["RANK"]) == 0:
                    (folder / "model_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            execution_state.update(update=step, training_active=True, unit_intervals=[], unit_gpu_events=[], unit_index=0)
            execution_state['optimizer_memory_diagnostic'] = settings.get('two_dimensional', {}).get('optimizer_memory_diagnostic', False)
            execution_state['joint_optimizer_cache_bootstrap'] = (settings.get('two_dimensional', {}).get('mode') == 'joint' or settings.get('native_n', False))
            accumulation.begin(step, kwargs.get("optimizer", args[3] if len(args) > 3 else None), model_chunks)
            numeric.step = step
            numeric.install(
                kwargs.get("optimizer", args[3] if len(args) > 3 else None),
                model_chunks,
            )
            completed_update_sync = hbm_guard_enabled
            if completed_update_sync and step==1:
                execution_state['params_ready_update']=0
                for chunk in model_chunks:
                    if getattr(chunk.config,'fp8',None) or getattr(chunk.config,'reuse_grad_buf_for_mxfp8_param_ag',False):
                        raise ValueError('Runtime capacity synchronous update boundary requires ordinary BF16 buffers')
                    original_zero=chunk.zero_grad_buffer
                    def runtime_zero(*zero_args,_original=original_zero,_chunk=chunk,**zero_kwargs):
                        result=_original(*zero_args,**zero_kwargs)
                        if execution_state.get('params_ready_update')==execution_state['update']-1 and execution_state['update']>1:
                            for group in _chunk.bucket_groups+_chunk.expert_parallel_bucket_groups:
                                if group.param_gather_handle is not None:raise RuntimeError('Parameter gather still pending at next update')
                                group.param_gather_dispatched=True
                        return result
                    chunk.zero_grad_buffer=runtime_zero
            logical.begin(model_chunks, step)
            # Frozen fixed-input plans need one exact input audit; keep its GPU
            # copies out of steady-state C/S timings. Streams audit every step,
            # as all N/C/S stream paths already do under the historical protocol.
            check_inputs = integrity is not None and (packet_source is not None or bool(settings.get("input_stream")) or step == 1)
            if check_inputs:
                integrity.begin(model_chunks, step)
            if communication is not None:
                communication.begin(step)
            operator_calls = {}
            audit_handles = []
            if step == 1 and settings.get("mixed_cp"):
                core = model_chunks[0]
                while hasattr(core, "module"):
                    core = core.module

                def count_call(name):
                    def hook(module, inputs, output):
                        operator_calls[name] += 1
                    return hook

                for index, layer in enumerate(core.decoder.layers):
                    modules = {"mlp": layer.mlp}
                    for name in ("linear_qkv", "linear_proj", "in_proj", "out_proj"):
                        module = getattr(layer.self_attention, name, None)
                        if module is not None:
                            modules[name] = module
                    for name, module in modules.items():
                        key = f"layer{index}.{name}"
                        operator_calls[key] = 0
                        audit_handles.append(module.register_forward_hook(count_call(key)))
            torch.cuda.synchronize()
            if hbm_guard_enabled:
                torch.distributed.barrier()
            torch.cuda.reset_peak_memory_stats()
            if hbm_guard_enabled:
                free_before, total_memory = torch.cuda.mem_get_info()
                packet_external_before = max(0, total_memory - free_before - torch.cuda.memory_reserved())
            start = time.perf_counter()
            try:
                value = original_step(*args, **kwargs)
                if completed_update_sync:
                    optimizer=kwargs.get('optimizer',args[3] if len(args)>3 else None)
                    with torch.no_grad():training.force_param_sync(model_chunks,optimizer=optimizer)
                    execution_state['params_ready_update']=step
                execution_state["training_active"] = False
                if settings.get("accumulation_units", 1) > 1:
                    from megatron.training import get_args
                    get_args().consumed_train_samples += (settings["accumulation_units"]-1)*len(execution_state["lengths"])
            finally:
                for handle in audit_handles:
                    handle.remove()
            torch.cuda.synchronize()
            if hbm_guard_enabled:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            rank_local_elapsed_ms = elapsed_ms
            if hbm_guard_enabled:
                completed_window = torch.tensor(elapsed_ms, device='cuda', dtype=torch.float64)
                torch.distributed.all_reduce(completed_window, op=torch.distributed.ReduceOp.MAX)
                elapsed_ms = completed_window.item()
            if operator_calls:
                expected_calls = len(execution_state["execution_rounds"]) * settings.get("accumulation_units", 1)
                if any(count != expected_calls for count in operator_calls.values()):
                    raise ValueError(f"Tokenwise operators were not fused per microbatch: {operator_calls}")
            if settings.get("two_dimensional", {}).get("mode") == "joint":
                if joint_position != {"update": step, "ordinal": len(joint_plans[step - 1].microbatches)}:
                    raise ValueError("Joint executor did not consume exactly one complete update")
            units = settings.get("accumulation_units", 1)
            if len(execution_state["unit_intervals"]) < units:
                # The last yielded batch is consumed without resuming the generator.
                unit = units - 1
                n = len(execution_state.get("lengths", {}))
                execution_state["unit_intervals"].append(dict(unit=unit,
                    parameter_version=step-1, begin_ns=execution_state.get("unit_start_ns", int(start*1e9)),
                    end_ns=time.perf_counter_ns(), logical_sample_ids=(execution_state["logical_sample_ids"] if packet_source is not None else list(range(((step-1)*units+unit)*n,((step-1)*units+unit+1)*n)))))
            from stream_replay import finite_result
            finite = finite_result(value)
            if not finite:
                raise RuntimeError("Nonfinite training loss or gradient norm")
            record = dict(
                finite=finite,
                losses={k: float(v.detach().float().item()) if torch.is_tensor(v) else float(v) for k, v in value[0].items()},
                grad_norm=float(value[5]) if value[5] is not None else None,
                input_lengths=list(execution_state.get("lengths", {}).values()),
                step=step,
                elapsed_ms=elapsed_ms,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                result=str(value),
                fingerprint=execution_state.get("fingerprint"),
                samples=len(execution_state.get("lengths", {})),
                tokens=sum(execution_state.get("lengths", {}).values()) * settings.get("accumulation_units", 1),
                accumulation_units=settings.get("accumulation_units", 1),
                tokens_per_unit=sum(execution_state.get("lengths", {}).values()),
                execution_microbatches_per_update=len(execution_state.get("execution_rounds", execution_state.get("plan", []))) * settings.get("accumulation_units", 1),
                sync=accumulation.finish(),
                unit_intervals=execution_state.get("unit_intervals", []),
                unit_compute_stream_ms=[a.elapsed_time(b) if b is not None else None for a,b in execution_state["unit_gpu_events"]],
                unit_timing_scope="Diagnostic compute-stream CUDA event F/B incl dependencies; pure compute and communication exposure unknown",
                microbatches=len(execution_state.get("execution_rounds", execution_state.get("plan", []))),
                source_microbatches=len(execution_state.get("plan", [])),
                mixed_cp=settings.get("mixed_cp", False),
                input_transport=input_transport,
                first_step_operator_calls=operator_calls or None,
            )
            if settings.get('two_dimensional_input'):
                record.update(parameter_sync_completed=True, rank_local_elapsed_ms=rank_local_elapsed_ms,
                    packet_scope='all_samples_in_one_frozen_update', loss_mask_contract='first_half_prefix_per_sample',
                    timing_contract='rank-max coordinated complete window: input delivery, F/B, gradient finalize, clip, Adam, updated-parameter all-gather and completion barrier; gathered parameters preserved at next gradient reset')
            if hbm_guard_enabled:
                free_after, total_memory = torch.cuda.mem_get_info()
                packet_external_after = max(0, total_memory - free_after - torch.cuda.memory_reserved())
                packet_budget = min(int(0.9 * total_memory), total_memory - 8 * 1024**3)
                packet_required = record['peak_allocated'] + max(packet_external_before, packet_external_after)
                admitted = packet_required <= packet_budget
                admission_flag = torch.tensor(int(admitted), device='cuda', dtype=torch.int32)
                torch.distributed.all_reduce(admission_flag, op=torch.distributed.ReduceOp.MIN)
                record.update(memory_budget_bytes=packet_budget, memory_external_before_bytes=packet_external_before,
                    memory_external_after_bytes=packet_external_after, memory_required_bytes=packet_required,
                    memory_admitted=admitted, all_ranks_memory_admitted=bool(admission_flag.item()),
                    memory_status='BUDGET_ACCEPTED' if admitted else 'BUDGET_REJECTED',
                    memory_check_scope='peak allocated plus larger before/after external occupancy, total-free-reserved; conservative boundary estimate, external occupancy separately recorded; allocated and reserved never added')
            if packet_source is not None:
                if value[1] != 0:
                    raise RuntimeError("Packet optimizer update skipped")
                execution_state["parameter_version"] = step
                from math import ceil
                lower = ceil(record["tokens"]/(int(os.environ["WORLD_SIZE"])*settings["packet_execution_cap"])) if settings.get("two_dimensional_input") else (32)
                if record["microbatches"] < lower or record["tokens"] != packet_source.manifest["streams"][0]["updates"][step-1]["input_tokens"]:
                    raise ValueError("Packet global batch capacity/token contract failed")
                record.update(recipe_units=None if settings.get('two_dimensional_input') else (32),consumed_train_samples_unit="packet",
                    cumulative_logical_samples=sum(len(v) for v in packet_source.samples[:step]),
                    cumulative_input_tokens=sum(u["input_tokens"] for u in packet_source.manifest["streams"][0]["updates"][:step]),packet_lookup_ms=execution_state["packet_lookup_ms"],
                    logical_sample_ids=execution_state["logical_sample_ids"],skipped_updates=int(value[1]))
                if settings.get('two_dimensional_input') and record['sync']['effective_loss_tokens'] != packet_source.manifest['loss_tokens']:
                    raise ValueError('Packet must retain the declared prefix loss tokens')
            record.update(
                samples_per_unit=len(execution_state.get("lengths", {})),
                samples=len(execution_state.get("lengths", {}))*settings.get("accumulation_units",1),
                source_microbatches_per_unit=record["source_microbatches"],
                execution_microbatches_per_unit=record["microbatches"],
                effective_loss_tokens=record["sync"]["effective_loss_tokens"],
                optimizer_updates=record["sync"]["optimizer_updates"],
                parameter_version=step-1,
            )
            if check_inputs:
                record["input_validation_ms"] = integrity.finish()
                record["input_integrity"] = ("passed_tokens_positions_labels_mask_exact" if packet_source is None or integrity.heavy else "passed_metadata_coverage_fb")
                if packet_source is not None:
                    record["packet_fb"] = execution_state["packet_fb"]
            folder = Path(os.environ["SEQUENCE_CP_OUTPUT"])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"steps_rank{os.environ['RANK']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if hbm_guard_enabled and not record['all_ranks_memory_admitted']:
                raise RuntimeError('Complete-update HBM budget rejected on at least one rank; all rank records preserved')
            logical.finish()
            numeric.finish_step()
            if communication is not None:
                communication.finish()
            return value

        training.train_step = train_step
