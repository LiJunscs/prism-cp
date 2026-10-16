"""Domain-checked CP1 cost query. Unsupported CP/topology never scales silently."""
import json
from pathlib import Path


class CostModel:
    def __init__(self, path):
        self.data=json.loads(Path(path).read_text())

    def query(self, module, phase, implementation, cp, topology, local_tokens, lengths):
        if cp!=1:
            raise ValueError('Uncalibrated CP; local Q/global KV and backward communication require real CP measurements')
        if not lengths or any(n<=0 for n in lengths) or sum(lengths)!=local_tokens:
            raise ValueError('CP1 lengths must be positive and sum to local_tokens')
        matches=[m for m in self.data['models'] if (m['module'],m['phase'],m['implementation'],m['cp'],m['topology']) ==
                 (module,phase,implementation,cp,topology)]
        if len(matches)!=1:
            raise ValueError('No matching module/phase/implementation/topology calibration')
        model=matches[0]
        exact=[r for r in self.data['measurements'] if r['module']==module and r['phase']==phase
               and r['implementation']==implementation and r['cp']==cp and r['topology']==topology and r['lengths']==list(lengths)]
        if exact:
            r=exact[0]
            return dict(milliseconds=r['median_ms'],interval_ms=[r['launch_min_ms'],r['launch_max_ms']],
                        peak_allocated_bytes=r['peak_allocated_bytes'],provenance='measured',case=r['case'])
        features=dict(s1_ki=local_tokens/1024,s2_ki2=sum(n*n for n in lengths)/1024**2,intercept=1)
        checks={**{k:features[k] for k in ('s1_ki','s2_ki2')},'sequence_count':len(lengths)}
        domain=model['domain']
        if any(not(domain[k][0]<=value<=domain[k][1]) for k,value in checks.items()) or any(
                not(domain['sequence_tokens'][0]<=n<=domain['sequence_tokens'][1]) for n in lengths):
            raise ValueError('Out of calibration domain; supplement measurements before scheduling')
        predicted=sum(features[k]*v for k,v in model['coefficients'].items())
        return dict(milliseconds=predicted,provenance='fitted',
                    reliable=model['engineering_target_pass'],relative_error_envelope=model['max_relative_error'],
                    peak_allocated_bytes=None,shape_support='feature box only; unmeasured packing')
