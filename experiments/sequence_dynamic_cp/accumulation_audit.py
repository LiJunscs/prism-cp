"""Record actual MCore bucket collective aliases; never infer physical link bytes."""
import functools
import json
import os
import time


class AccumulationAudit:
    def __init__(self, state, output):
        self.state, self.output = state, output
        self.active = False
        self.installed = False
        self.calls = []
        self.trace = os.environ.get("SEQUENCE_CP_COMM_AUDIT") == "1"
        self.diagnostic = False
        self.timeline = []

    def begin(self, update, optimizer, models):
        self.active = True
        self.diagnostic = self.trace and update in (6,7)
        self.calls = []
        self.timeline = []
        self.optimizer_steps = 0
        self.finalizes = 0
        self.loss_tokens = None
        self.update = update
        if self.installed:
            return
        self.installed = True
        import torch
        def memory(stage):
            if not self.state.get('optimizer_memory_diagnostic'):
                return
            free, total = torch.cuda.mem_get_info()
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output/f'optimizer_memory_rank{os.environ["RANK"]}.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(update=self.update, stage=stage,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    free=free, total=total))+'\n')
        from megatron.core.distributed import param_and_grad_buffer as bucket_module
        from megatron.core.utils import get_model_config
        # These aliases are invoked inside the actual per-bucket loop, including
        # direct _reduce_scatter_base/_all_gather_base selections by MCore.
        for name, op in [('dist_reduce_scatter_func', 'RS'), ('dist_all_gather_func', 'AG')]:
            original = getattr(bucket_module, name)
            def wrap(fn, op):
                @functools.wraps(fn)
                def call(output, input, *args, **kwargs):
                    if self.active:
                        unit = self.state.get('unit_index', 0)
                        if op == 'RS' and unit != self.state.get('accumulation_units', 1)-1:
                            raise RuntimeError('Gradient reduction before final accumulation unit')
                        packet = self.state.get('packet_integrity')
                        if packet and op == 'RS':
                            count = len(self.state['execution_rounds'])
                            if packet.ordinal != count or packet.backwards < count-1:
                                raise RuntimeError('Gradient RS before final packet backward')
                        group = kwargs.get('group')
                        self.calls.append(dict(update=self.update, unit=unit, op=op,
                            input_bytes=input.numel()*input.element_size(),
                            output_bytes=output.numel()*output.element_size(),
                            payload_bytes=(input if op=='RS' else output).numel()*(input if op=='RS' else output).element_size(),
                            group_ranks=torch.distributed.get_process_group_ranks(group),
                            host_dispatch_ns=time.perf_counter_ns(), completion_ns='unknown',
                            packet_forwards=packet.ordinal if packet else None,
                            packet_backwards_complete=packet.backwards if packet else None,
                            packet_stage='last_backward_or_finalize' if packet and op=='RS' else None))
                    return fn(output, input, *args, **kwargs)
                return call
            setattr(bucket_module, name, wrap(original, op))
        if self.trace:
            cls = bucket_module._ParamAndGradBucketGroup
            for name in ('register_grad_ready', 'start_grad_sync', 'start_param_sync', 'finish_grad_sync', 'finish_param_sync'):
                original = getattr(cls, name)
                def method(fn, name):
                    @functools.wraps(fn)
                    def invoke(group, *args, **kwargs):
                        relevant = self.active and self.diagnostic and (name != 'register_grad_ready' or group.is_last_microbatch)
                        row = None
                        if relevant:
                            row = dict(update=self.update, unit=self.state.get('unit_index',0),
                                action=name, bucket_group_id=id(group),
                                bucket_indices=[getattr(b,'bucket_id',i) for i,b in enumerate(group.buckets)],
                                host_begin_ns=time.perf_counter_ns(),
                                scope='Host readiness/dispatch/wait callback; CUDA marker after wait establishes dependency completion on compute stream, not isolated NCCL kernel duration')
                            if name=='register_grad_ready':
                                row['parameter_id']=id(args[0] if args else kwargs['param'])
                        result=fn(group,*args,**kwargs)
                        if row is not None:
                            row['host_return_ns']=time.perf_counter_ns()
                            event=None
                            if name.startswith('finish_') and self.state.get('unit_gpu_events'):
                                event=torch.cuda.Event(enable_timing=True);event.record()
                            self.timeline.append((row,event))
                        return result
                    return invoke
                setattr(cls,name,method(original,name))
        original_step = optimizer.step
        def step(*args, **kwargs):
            memory('before_optimizer')
            if self.active and self.state.get('packet_integrity'):
                audit = self.state['packet_integrity']
                expected = len(self.state['execution_rounds'])
                if audit.ordinal != expected or audit.backwards != expected or self.finalizes != 1:
                    raise RuntimeError('Optimizer invoked before all packet F/B and finalization')
            self.optimizer_steps += int(self.active)
            begin=time.perf_counter_ns()
            if self.active and self.diagnostic:
                with torch.cuda.nvtx.range(f'optimizer/update={self.update}'):
                    result=original_step(*args, **kwargs)
            else:
                result=original_step(*args, **kwargs)
            memory('after_optimizer')
            if self.active and self.diagnostic:
                self.timeline.append((dict(update=self.update,unit=self.state.get('unit_index',0),
                    action='optimizer_step',host_begin_ns=begin,host_return_ns=time.perf_counter_ns(),
                    scope='Host optimizer interval, includes clip/update/preparation; CUDA execution and param-prefetch exposure require raw timeline'),None))
            return result
        optimizer.step = step
        config = get_model_config(models[0])
        original_finalize = config.finalize_model_grads_func
        def finalize(models, num_tokens, *args, **kwargs):
            memory('before_finalize')
            if self.update == 1 and self.state.get('joint_optimizer_cache_bootstrap'):
                # F/B has freed its activations but their cached blocks can
                # starve NCCL's first finalization/statistics allocations, which
                # bypass PyTorch's allocator. Release unused blocks once during
                # bootstrap; keep every gradient collective, norm and Adam step.
                torch.cuda.empty_cache()
                memory('after_bootstrap_cache_release')
            if self.active and self.state.get('packet_integrity'):
                audit = self.state['packet_integrity']
                if audit.backwards != len(self.state['execution_rounds']):
                    raise RuntimeError('Finalization before all packet backwards')
            self.finalizes += int(self.active)
            if self.active and self.state.get('unit_gpu_events'):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.state['unit_gpu_events'][-1][1] = end
                torch.cuda.nvtx.mark(f'accum_unit_end/update={self.update}/unit={self.state.get("unit_index",0)}')
            result = original_finalize(models, num_tokens, *args, **kwargs)
            memory('after_finalize')
            # finalize all-reduces loss-mask token count over DPxCP in place.
            if self.active:
                self.loss_tokens = num_tokens.detach() if self.state.get('packet_integrity') else int(num_tokens.item())
            return result
        config.finalize_model_grads_func = finalize

    def finish(self):
        self.active = False
        if self.optimizer_steps != 1 or self.finalizes != 1:
            raise RuntimeError(f'Expected one optimizer/finalize: {self.optimizer_steps}/{self.finalizes}')
        if hasattr(self.loss_tokens, 'item'):
            self.loss_tokens = int(self.loss_tokens.item())
        expected = sum(self.state['lengths'].values())//2*self.state.get('accumulation_units', 1)
        if self.loss_tokens != expected:
            raise RuntimeError(f'Loss-token normalization differs: {self.loss_tokens} vs {expected}')
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/f'buckets_rank{os.environ["RANK"]}.jsonl').open('a') as handle:
            for row in self.calls:
                handle.write(json.dumps(row)+'\n')
        if self.diagnostic:
            origin=self.state['unit_gpu_events'][0][0] if self.state.get('unit_gpu_events') else None
            with (self.output/f'timeline_rank{os.environ["RANK"]}.jsonl').open('a') as handle:
                for row,event in self.timeline:
                    row['compute_stream_dependency_marker_ms_from_first_unit']=origin.elapsed_time(event) if origin is not None and event is not None else 'unknown'
                    handle.write(json.dumps(row)+'\n')
        return dict(optimizer_updates=self.optimizer_steps, finalize_calls=self.finalizes,
            effective_loss_tokens=self.loss_tokens,
            RS_buckets=sum(c['op']=='RS' for c in self.calls),
            AG_buckets=sum(c['op']=='AG' for c in self.calls),
            RS_payload_bytes=sum(c['payload_bytes'] for c in self.calls if c['op']=='RS'),
            AG_payload_bytes=sum(c['payload_bytes'] for c in self.calls if c['op']=='AG'),
            scope='Actual MCore per-bucket API dispatch; not physical cross-node bytes; AG may prefetch following update')
