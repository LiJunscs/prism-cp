"""Trace-only Python collective payload inventory; no physical link timing claims."""
import functools
import json
import traceback
from pathlib import Path


class CommunicationAudit:
    def __init__(self, output: Path, target_step: int):
        import torch.distributed as dist
        self.output, self.target_step = output, target_step
        self.active = False
        self.records = []
        positions = {'all_gather_into_tensor': 2, 'all_gather': 2, 'all_to_all_single': 4,
                     'all_to_all': 2, 'all_reduce': 2, 'reduce_scatter_tensor': 3,
                     'reduce_scatter': 3}
        for name, group_index in positions.items():
            original = getattr(dist, name)

            @functools.wraps(original)
            def wrapped(*args, _name=name, _group_index=group_index, _original=original, **kwargs):
                if self.active:
                    group = kwargs.get('group', args[_group_index] if len(args) > _group_index else None)
                    self.record(_name, [*args[:2], *[kwargs[k] for k in ('tensor','input','output','input_tensor','output_tensor','input_tensor_list','output_tensor_list') if k in kwargs]], group)
                return _original(*args, **kwargs)

            setattr(dist, name, wrapped)
        original_batch = dist.batch_isend_irecv

        @functools.wraps(original_batch)
        def batch(ops):
            if self.active:
                for op in ops:
                    self.record(op.op.__name__, [op.tensor], op.group, peer=op.peer)
            return original_batch(ops)

        dist.batch_isend_irecv = batch

    def record(self, name, values, group, peer=None):
        import torch
        import torch.distributed as dist
        tensors = []
        def add(value):
            if torch.is_tensor(value):
                tensors.append(dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device), bytes=value.numel()*value.element_size()))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    add(child)
        for value in values:
            add(value)
        pg = group if group is not None else dist.group.WORLD
        self.records.append(dict(operation=name, rank=dist.get_rank(), group_ranks=dist.get_process_group_ranks(pg), peer=peer, tensors=tensors, stack=[f'{f.filename}:{f.lineno}:{f.name}' for f in traceback.extract_stack(limit=9)[:-1]], scope='Python API tensor shapes/bytes; not physical wire traffic or completed link duration'))

    def begin(self, step):
        self.active = step == self.target_step

    def finish(self):
        if not self.active:
            return
        import torch.distributed as dist
        self.active = False
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/f'communication_rank{dist.get_rank()}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))
