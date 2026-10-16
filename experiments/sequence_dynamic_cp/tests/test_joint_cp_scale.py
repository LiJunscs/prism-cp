"""Unmeasured CP32 must be explicit; memory capacity is a necessary bound."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_cp import Group, Layout, Microbatch, Model, Plan, SearchOptions, solve, validate_plan
from joint_cp.model import Evaluator
from joint_cp.response import load_response
from joint_cp_scale_audit import add_template, count_dp


def cost(model, symbol, cp, lengths=(65536,4096)):
    ls=dict(enumerate(lengths))
    layout=Layout((Group(tuple(range(cp)),tuple(ls)),),"zigzag")
    return Evaluator(model,ls,cp,"GGG*GGG*").operator(symbol,layout)[0]


def test_cp32_needs_explicit_hypothesis_and_does_not_modify_cp16():
    original=Model(cp_response=load_response())
    with pytest.raises(ValueError,match="explicit"):
        cost(original,"*",32)
    times=[]
    for policy in ("work_scaling","carry_residual","double_residual"):
        m=Model(cp_response=load_response(cp32=policy))
        times.append(cost(m,"*",32))
        assert cost(m,"*",16)==pytest.approx(cost(original,"*",16))
        assert cost(m,"G",16)==pytest.approx(cost(original,"G",16))
    assert times[0] < times[1] < times[2]


def test_cp32_gdn_stress_preserves_other_operators_and_cp_sizes():
    base=Model(cp_response=load_response(cp32="carry_residual"))
    stress=Model(cp_response=load_response(cp32="carry_residual",gdn32_multiplier=1.5))
    assert cost(stress,"G",32)==pytest.approx(1.5*cost(base,"G",32))
    assert cost(stress,"M",32)==cost(base,"M",32)
    assert cost(stress,"*",32)==cost(base,"*",32)
    assert cost(stress,"G",16)==cost(base,"G",16)
    with pytest.raises(ValueError,match="multiplier"):
        Model(cp_response=load_response(cp32="carry_residual",gdn32_multiplier=.5))


def test_necessary_hbm_ceiling_blocks_two_long_samples_on_16_but_not_32():
    m=Model(cp_response=load_response(cp32="carry_residual"))
    assert m.token_capacity_upper_bound("GGG*GGG*",16)==487840
    assert m.token_capacity_upper_bound("GGG*GGG*",32)==990105
    lengths={0:262144,1:262144}
    assert sum(lengths.values()) > m.token_capacity_upper_bound("GGG*GGG*",16)
    for world, fits in ((16,False),(32,True)):
        qa=Layout((Group(tuple(range(world)),(0,1)),),"zigzag")
        gd=qa.with_mode("contiguous")
        batch=Microbatch(qa,tuple((gd,gd) if c=="G" else (qa,qa) for c in "GGG*GGG*"))
        ev=Evaluator(m,lengths,world,"GGG*GGG*")
        assert ev.evaluate(batch)["fits"] is fits
    assert replace(m,device_bytes=90*1024**3).token_capacity_upper_bound("GGG*GGG*",16) > 524288


def test_32_rank_cp16_control_does_not_call_unmeasured_attention_cp32():
    m=Model(cp_response=load_response())
    result=solve({0:65536,1:4096},32,"G*",m,SearchOptions(beam=8,layouts=6,max_attention_cp=16))
    plan=Plan.from_dict(result["plan"])
    validate_plan(plan,{0:65536,1:4096})
    assert all(len(g.ranks)<=16 for b in plan.microbatches for a,_ in b.layers for g in a.groups)
    assert result["cp_response_evidence"]["counts"]["cp32_gqa_calls"]==0
    with pytest.raises(ValueError,match="ceiling"):
        SearchOptions(max_attention_cp=12)


def test_count_dp_freely_reallocates_shorts_and_requires_complete_coverage():
    # A single repeated heavy+short recipe costs 16, whereas two different
    # microbatches cover the same global pool in 13. Greedy per-batch selection
    # also fails because it leaves an impossible short remainder.
    rows=[(1,0,5.,0),(1,1,8.,1),(1,2,8.,2),(0,2,20.,3)]
    score,chosen=count_dp(rows,2,2)
    assert score==13.
    assert sorted(chosen)==[0,2]
    assert sum(rows[i][0] for i in chosen)==2
    assert sum(rows[i][1] for i in chosen)==2
    with pytest.raises(ValueError,match="complete input"):
        count_dp([(1,2,8.,0)],2,3)


def test_catalogue_identity_includes_lengths_as_well_as_ownership():
    qa=Layout((Group((0,1),(0,)),),"zigzag")
    batch=Microbatch(qa,tuple((qa.with_mode("contiguous" if c=="G" else "zigzag"),qa)
                             for c in "GGG*GGG*"))
    catalogue={}
    add_template(catalogue,batch,{0:16},16,4,2,"long")
    add_template(catalogue,batch,{0:4},16,4,2,"short")
    add_template(catalogue,batch,{0:4},16,4,2,"duplicate")
    assert len(catalogue)==2
    assert {(r['long_samples'],r['short_samples']) for r in catalogue.values()}=={(1,0),(0,1)}
