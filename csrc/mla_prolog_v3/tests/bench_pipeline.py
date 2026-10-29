"""GLM baseline/candidate harness (BF16 by default, MODE=mxfp8); run on an idle visible device.

LIB=... OUT=... TOKENS=1,32,92,32768 python3 bench_pipeline.py
REF=... compares deterministic output rows (all rows for decode) bit for bit.
Graph batches isolate device time from Python dispatch; events report us/op.
"""
import gc
import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu

root = Path(__file__).resolve().parents[3]
lib = os.environ.get('LIB', str(root / 'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so'))
# Reuse the independently golden-checked MXFP8 input construction.
source = Path(__file__).with_name('test_mxfp8_precision.py').read_text()
source = source[:source.index('# golden on')]
source = source.replace('torch.ops.load_library(LIB)', 'torch.ops.load_library(os.environ.get("LIB", LIB))')
source = source.replace('query_norm_flag=False', 'query_norm_flag=True')
source = source.replace('q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(', 'def call():\n    return torch.ops.npu.mla_prolog_v3(')
source = source.replace('torch_npu.npu.synchronize()', '')
if os.environ.get('MODE', 'bf16') == 'bf16':
    source = source.replace('.to(torch.float8_e4m3fn)', '')
    source = source.replace('weight_quant_mode=3', 'weight_quant_mode=0')
    source = source.replace('    dequant_scale_x=sx, dequant_scale_w_dq=sdq, dequant_scale_w_uq_qr=suq, dequant_scale_w_dkv_kr=sdkv)', ')')
out = Path(os.environ.get('OUT', '/tmp/mla_v3_pipeline'))
out.mkdir(parents=True, exist_ok=True)
results = []
for t in map(int, os.environ.get('TOKENS', '1,32,92,8192,30456,32768').split(',')):
    os.environ['T'] = str(t)
    ns = {'__name__': '__pipeline_inputs__'}
    exec(compile(source, 'pipeline_inputs', 'exec'), ns)
    call = ns['call']
    for _ in range(5):
        got = call()
    torch.npu.synchronize()
    rows = torch.arange(t) if t <= 256 else torch.unique(torch.cat((torch.arange(128), torch.arange(t-128,t), torch.linspace(0,t-1,512).long())))
    samples = [v.float()[rows.to(v.device)].cpu() if v is not None and v.numel() and v.shape[0] == t else (v.cpu() if v is not None else None) for v in got]
    for key in ('kvc', 'krc'):
        samples.append(ns[key].view(-1, ns[key].shape[-1])[rows.to(ns[key].device)].cpu())
    ref = os.environ.get('REF')
    diffs = []
    if ref:
        expected = torch.load(Path(ref) / f'T{t}.pt', weights_only=True)
        for a,b in zip(samples, expected):
            if a is None or b is None:
                assert a is b
                diffs.append(0.)
            else:
                diff = (a.float()-b.float()).abs().max().item() if a.numel() else 0.
                diffs.append(diff)
                cos = torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), 0).item() if a.numel() else 1.
                if os.environ.get("CHECK", "bitwise") == "bitwise":
                    assert torch.equal(a,b), (t, diffs, cos)
                else:
                    print("precision", t, len(diffs)-1, "max_abs", diff, "cosine", cos, flush=True)
                    assert cos > 0.99999, (t, diffs, cos)
    torch.save(samples, out / f'T{t}.pt')
    graph = torch.npu.NPUGraph()
    n = 5
    with torch.npu.graph(graph):
        for _ in range(n):
            got = call()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(21)]
    for a,b in events:
        a.record()
        graph.replay()
        b.record()
    torch.npu.synchronize()
    times = [a.elapsed_time(b)*1000/n for a,b in events]
    result = dict(tokens=t, mode=os.environ.get('MODE', 'bf16'), median_us=statistics.median(times), min_us=min(times), max_us=max(times), max_abs_diffs=diffs)
    results.append(result)
    print(json.dumps(result), flush=True)
    (out/'timing.json').write_text(json.dumps(results, indent=2))
    del graph, got, call, ns, samples
    gc.collect()
    torch.npu.empty_cache()
