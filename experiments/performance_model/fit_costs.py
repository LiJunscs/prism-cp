"""Fit the prespecified nonnegative P1 model; holdout never selects the model."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from itertools import combinations

import numpy as np


def nnls(matrix, target):
    """Exact active-set enumeration for these two/three-feature models.

    Each feasible optimum lies on a face of the nonnegative orthant. Solve
    unconstrained least squares on every face, retaining only feasible fits.
    This small model needs at most eight solves and no scipy dependency.
    """
    width = matrix.shape[1]
    best = np.zeros(width)
    error = float(np.linalg.norm(target))
    for size in range(1, width + 1):
        for active in combinations(range(width), size):
            values = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
            if np.any(values < -1e-12):
                continue
            candidate = np.zeros(width)
            candidate[list(active)] = np.maximum(values, 0)
            residual = float(np.linalg.norm(matrix @ candidate - target))
            if residual < error:
                best, error = candidate, residual
    return best, error


def fit(directory):
    with (directory/'measurements.csv').open() as f:
        raw = [r for r in csv.DictReader(f) if r['status']=='success']
    grouped = {}
    for row in raw:
        key = (row['module'],row['phase'],row['case'],row['launch'])
        grouped.setdefault(key,[]).append(row)
    launches = []
    for (module,phase,case,launch),rows in grouped.items():
        times=[float(r['milliseconds']) for r in rows]
        r=rows[0]
        launches.append(dict(module=module,phase=phase,case=case,launch=launch,
            split=r['split'],s1_ki=float(r['s1_ki']),s2_ki2=float(r['s2_ki2']),
            lengths=json.loads(r['lengths']),implementation=r['implementation'],cp=int(r['cp']),topology=r['topology'],
            median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
            p10_ms=float(np.percentile(times,10)),p90_ms=float(np.percentile(times,90)),
            iterations=len(times),peak_allocated_bytes=max(int(x['allocated_bytes']) for x in rows),
            peak_reserved_bytes=max(int(x['reserved_bytes']) for x in rows)))
    points={}
    for row in launches:
        points.setdefault((row['module'],row['phase'],row['case']),[]).append(row)
    aggregates=[]
    for key, rows in points.items():
        r=rows[0].copy()
        r.pop('launch');r.pop('iterations')
        medians=[x['median_ms'] for x in rows]
        r.update(median_ms=statistics.median(medians),launch_min_ms=min(medians),
            launch_max_ms=max(medians),independent_launches=len(rows),
            peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows),
            peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rows))
        aggregates.append(r)
    models=[]; residuals=[]
    for module in sorted({r['module'] for r in aggregates}):
        for phase in ('forward','backward','fwd_bwd'):
            rows=[r for r in aggregates if r['module']==module and r['phase']==phase]
            train=[r for r in rows if r['split']=='train']
            features=['s2_ki2','s1_ki','intercept'] if module in ('gqa','dsa','csa','csa128') else ['s1_ki','intercept']
            def design(r):
                return [1 if feature=='intercept' else r[feature] for feature in features]
            matrix=np.array([design(r) for r in train])
            if np.linalg.matrix_rank(matrix)<len(features):
                raise ValueError(f'{module}/{phase}: insufficient independent features')
            # Scale columns for NNLS conditioning; equal point weights, fixed a priori.
            scale=np.maximum(matrix.max(axis=0),1)
            parameters,_=nnls(matrix/scale,np.array([r['median_ms'] for r in train]))
            parameters=parameters/scale
            errors=[]; holdout_errors=[]
            for r in rows:
                predicted=float(np.dot(design(r),parameters))
                signed=predicted-r['median_ms']
                relative=abs(signed)/r['median_ms']
                errors.append(relative)
                if r['split']=='holdout':
                    holdout_errors.append(relative)
                residuals.append(dict(module=module,phase=phase,case=r['case'],split=r['split'],
                    measured_ms=r['median_ms'],predicted_ms=predicted,absolute_error_ms=abs(signed),
                    relative_error=relative,launch_min_ms=r['launch_min_ms'],launch_max_ms=r['launch_max_ms'],
                    independent_launches=r['independent_launches'],provenance='fitted_vs_measured'))
            models.append(dict(module=module,phase=phase,implementation=train[0]['implementation'],
                cp=train[0]['cp'],topology=train[0]['topology'],features=features,
                coefficients=dict(zip(features,parameters.tolist())),
                units=dict(time='ms',s1='Ki tokens',s2='(Ki tokens)^2'),
                method='NNLS equal weight per shape on median of independent launch medians',
                train_cases=[r['case'] for r in train],
                domain=dict(s1_ki=[min(r['s1_ki'] for r in train),max(r['s1_ki'] for r in train)],
                    s2_ki2=[min(r['s2_ki2'] for r in train),max(r['s2_ki2'] for r in train)],
                    sequence_tokens=[min(min(r['lengths']) for r in train),max(max(r['lengths']) for r in train)],
                    sequence_count=[min(len(r['lengths']) for r in train),max(len(r['lengths']) for r in train)]),
                max_relative_error=max(errors),holdout_median_relative_error=statistics.median(holdout_errors) if holdout_errors else None,
                holdout_all_under_20pct=all(e<=.2 for e in holdout_errors) if holdout_errors else None,
                engineering_target_pass=statistics.median(holdout_errors)<=.1 and sum(e<=.2 for e in holdout_errors)>len(holdout_errors)/2 if holdout_errors else False,
                provenance='fitted',source=str(directory/'measurements.csv')))
    costs=dict(schema_version=1,models=models,measurements=aggregates,
        uncertainty='empirical launch min/max, not independent confidence interval; residual is descriptive',
        limitations=['CP1 only; not a 1/p scaling model','finite calibration domain; in-domain feature box does not prove shape support',
            'fwd_bwd is fit separately and need not equal two fitted phases','no communication or overlap calibration'],
        failures=[str(p) for p in directory.glob('**/failures.csv')])
    (directory/'costs.json').write_text(json.dumps(costs,indent=2))
    for filename, rows in [('launch_summary.csv',launches),('fit_residuals.csv',residuals)]:
        with (directory/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for m in models:
        error=m['holdout_median_relative_error']
        print(m['module'],m['phase'],m['coefficients'],'holdout median:',error,'target:',m['engineering_target_pass'])
    return costs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    fit(parser.parse_args().directory)
