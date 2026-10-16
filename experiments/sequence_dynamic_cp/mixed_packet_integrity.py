"""Warmup exact input checks; measurement retains only coverage and counters."""
import os
import time
from stream_integrity import StreamIntegrity
from mixed_global_batch import fields


class PacketIntegrity(StreamIntegrity):
    def begin(self, model, step):
        self.heavy = step <= self.state['packet_warmup']
        self.flags = []
        self.references = {}
        self.seen = {}
        self.backwards = 0
        self.host_check_ms = 0.
        super().begin(model, step)
        if not hasattr(self, 'backward_installed'):
            core = model[0]
            while hasattr(core, 'module'):
                core = core.module
            def forward(module, inputs, kwargs, output):
                if self.active:
                    if self.state['parameter_version'] != self.step-1:
                        raise ValueError('Parameter version changed inside packet')
                    output.register_hook(self.backward)
            core.embedding.register_forward_hook(forward, with_kwargs=True)
            if self.state.get('capacity_preflight'):
                def memory_forward(module,inputs,output):
                    if self.active:self.memory_boundary('F_end',self.ordinal)
                core.register_forward_hook(memory_forward)
                original_ce=core.compute_language_model_loss
                def memory_ce(*args,**kwargs):
                    if self.active:self.memory_boundary('CE_before',self.ordinal)
                    result=original_ce(*args,**kwargs)
                    if self.active:self.memory_boundary('CE_after',self.ordinal)
                    return result
                core.compute_language_model_loss=memory_ce
            self.backward_installed = True

    def backward(self, grad):
        if self.active:
            self.backwards += 1
            if self.state.get('capacity_preflight'):
                self.memory_boundary('B_end',self.backwards)
            if self.state['parameter_version'] != self.step-1:
                raise ValueError('Parameter version changed during backward')
        return grad

    def memory_boundary(self,phase,ordinal):
        if ordinal<=2 or ordinal%8==0 or (self.state.get("arch_capacity") and ordinal==len(self.state.get("execution_rounds",[]))):
            import torch
            self.state.setdefault('memory_boundary_samples',[]).append(dict(phase=phase,microbatch=ordinal,
                allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                active=torch.cuda.memory_stats().get('active_bytes.all.current',0),inactive_split=torch.cuda.memory_stats().get('inactive_split_bytes.all.current',0),free_total=list(torch.cuda.mem_get_info()),
                scope='small preflight F/B boundary sample, caching/optimizer allocations may persist'))

    def capture(self, name, value):
        import torch
        started = time.perf_counter()
        if value is None:
            raise ValueError('Missing packet field '+name)
        if name == 'tokens':
            self.references = {}
        flat = value.detach().reshape(-1)
        rank = int(os.environ['RANK'])
        cursor = 0
        for g in self.group:
            cp = len(g.ranks)
            local_rank = g.ranks.index(rank)
            for sample in g.ids:
                n = self.state['lengths'][sample]
                count = n//cp
                self.seen.setdefault(sample, []).append(name)
                if self.heavy:
                    if sample not in self.references:
                        width = n//(2*cp)
                        positions = torch.cat((torch.arange(local_rank*width,(local_rank+1)*width),
                            torch.arange(n-(local_rank+1)*width,n-local_rank*width)))
                        self.references[sample] = fields(n,self.state['logical_sample_ids'][sample],self.vocab,self.eod,
                                       device=value.device,positions=positions)
                    reference = self.references[sample]
                    reference_name = 'position_ids' if name=='positions' else name
                    self.flags.append((flat[cursor:cursor+count] == reference[reference_name]).all())
                cursor += count
        if name == 'loss_mask':
            self.references = {}
        if cursor != flat.numel():
            raise ValueError('Packet field token count mismatch')
        self.host_check_ms += (time.perf_counter()-started)*1000

    def finish(self):
        import torch
        start = time.perf_counter()
        self.active = False
        rank = int(os.environ['RANK'])
        expected = {i for row in self.state['plan'] for g in row if rank in g.ranks for i in g.ids}
        if set(self.seen) != expected or any(sorted(v)!=['labels','loss_mask','positions','tokens'] for v in self.seen.values()):
            raise ValueError('Packet rank sample coverage incomplete or duplicated')
        count = len(self.state['execution_rounds'])
        if self.ordinal != count or self.backwards != count:
            raise ValueError(f'Packet F/B mismatch {self.ordinal}/{self.backwards}/{count}')
        if self.heavy and (not self.flags or not bool(torch.stack(self.flags).all().item())):
            raise ValueError('Packet input content differs from logical-ID reference')
        self.flags = []; self.parts = []; self.references = {}
        self.state['packet_fb'] = dict(forwards=self.ordinal,backwards=self.backwards,
                                      parameter_version=self.step-1,coverage='exact_rank_ids',host_input_check_ms=self.host_check_ms,
                                      check_scope='Host input validation dispatch only; GPU work in warmup is included in full elapsed')
        return (time.perf_counter()-start)*1000
