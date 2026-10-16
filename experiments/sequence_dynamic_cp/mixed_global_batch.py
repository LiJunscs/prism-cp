"""Packet collation and reversible plan identities for frozen inputs."""

import base64
import json
import zlib


def structural_identity(plan, lengths):
    # Raw DEFLATE has no checksum. This is reversible encoding, not hashing.
    shape = [[dict(ranks=g.ranks, lengths=[lengths[i] for i in g.ids])
              for g in row] for row in plan]
    payload = json.dumps(shape, sort_keys=True, separators=(',', ':')).encode()
    coder = zlib.compressobj(level=6, wbits=-15)
    return 'structure_v1:' + base64.b64encode(coder.compress(payload)+coder.flush()).decode()


def install_identity():
    import planner
    planner.fingerprint = structural_identity


def fields(n, logical_id, vocab, eod, device='cpu', positions=None, seed=1234):
    import torch
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + logical_id)
    full = torch.randint(0, vocab, (n+1,), generator=generator, dtype=torch.int64)
    full[-1] = eod
    positions_cpu = torch.arange(n, dtype=torch.int64) if positions is None else positions.cpu()
    positions = positions_cpu.to(device)
    return dict(tokens=full[:-1][positions_cpu].to(device), labels=full[1:][positions_cpu].to(device),
                position_ids=positions, loss_mask=(positions >= n//2).float())


def packet_collate(items):
    if len(items) != 1 or not isinstance(items[0], list):
        raise ValueError('Packet loader requires exactly one packet per batch')
    return items[0]
