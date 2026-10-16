"""Exact input checks after the timed step; retain only four small input fields."""
import os
import sys
import time


def reference_fields(n, cp, local_rank, index, vocab, eod, device):
    import torch
    width=n//(2*cp)
    positions=torch.cat((torch.arange(local_rank*width,(local_rank+1)*width,device=device),torch.arange(n-(local_rank+1)*width,n-local_rank*width,device=device)))
    tokens=(positions*7+index*17+3)%vocab
    labels=((positions+1)*7+index*17+3)%vocab
    labels[positions==n-1]=eod
    return dict(tokens=tokens,positions=positions,labels=labels,loss_mask=(positions>=n//2).float())


class StreamIntegrity:
    def __init__(self,state,sample_count):
        self.state=state;self.sample_count=sample_count;self.installed=False;self.active=False

    def begin(self,model,step):
        self.active=True;self.step=step;self.ordinal=0;self.parts=[];self.group=None
        if self.installed:return
        from megatron.training import get_tokenizer
        tok=get_tokenizer();self.vocab=tok.vocab_size;self.eod=tok.eod
        core=model[0]
        while hasattr(core,'module'):core=core.module
        def embedding(module,inputs,kwargs,output):
            if not self.active:return
            rank=int(os.environ['RANK'])
            from mixed_plan import local_execution_groups
            self.group=local_execution_groups(self.state,self.ordinal % len(self.state.get("execution_rounds",self.state["plan"])),rank);self.ordinal+=1
            self.capture('tokens',kwargs.get('input_ids',inputs[0] if inputs else None))
            self.capture('positions',kwargs.get('position_ids',inputs[1] if len(inputs)>1 else None))
        core.embedding.register_forward_hook(embedding,with_kwargs=True)
        original_loss=core.compute_language_model_loss
        def loss(labels,logits):
            if self.active:self.capture('labels',labels)
            return original_loss(labels,logits)
        core.compute_language_model_loss=loss
        main=sys.modules['__main__'];original_masked=main.loss_func
        def masked(mask,output,*args,**kwargs):
            if self.active:self.capture('loss_mask',mask)
            return original_masked(mask,output,*args,**kwargs)
        main.loss_func=masked;self.installed=True

    def capture(self,name,value):
        if value is None:raise ValueError('Missing actual stream input '+name)
        rank=int(os.environ['RANK']);cursor=0
        flat=value.detach().reshape(-1)
        for g in self.group:
            cp=len(g.ranks)
            for sample in g.ids:
                n=self.state['lengths'][sample];count=n//cp
                self.parts.append((name,sample,self.state.get("unit_index",0),n,cp,g.ranks.index(rank),flat[cursor:cursor+count].clone()))
                cursor+=count
        if flat.numel()!=cursor:raise ValueError('Stream input row count mismatch')

    def finish(self):
        import torch
        start=time.perf_counter();self.active=False
        if self.ordinal!=len(self.state.get('execution_rounds',self.state['plan']))*self.state.get("accumulation_units",1):raise ValueError('Stream microbatch count mismatch')
        seen={};flags=[];expected={}
        for name,sample,unit,n,cp,local_rank,actual in self.parts:
            seen.setdefault((unit,sample),[]).append(name)
            key=(unit,sample,n,cp,local_rank)
            if key not in expected:expected[key]=reference_fields(n,cp,local_rank,((self.step-1)*self.state.get("accumulation_units",1)+unit)*(self.sample_count or len(self.state['lengths']))+sample,self.vocab,self.eod,actual.device)
            flags.append((actual==expected[key][name]).all())
        required={'tokens','positions','labels','loss_mask'}
        if not seen or any(len(names)!=4 or set(names)!=required for names in seen.values()):raise ValueError('Stream input fields incomplete or duplicated')
        passed=bool(torch.stack(flags).all().item());self.parts=[]
        if not passed:raise ValueError('Actual stream tokens/positions/labels/mask differ from deterministic native fixture')
        return (time.perf_counter()-start)*1000
