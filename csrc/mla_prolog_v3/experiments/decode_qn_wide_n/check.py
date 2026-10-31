"""Five-way decode ablation. Run only on an idle, unoccupied device.

The scripts compare each route with the original outputs before timing it.
Old: contiguous Query + concat; packed: direct 576 output; head/parallel/both
add internal-kernel experiments. All variants share one process and binary.
"""
import json
import os
import statistics
from pathlib import Path

# Reuse exactly the same input distribution, weights, layout guards and timing
# setup as the validated layout test, without executing its test loop.
exec((Path(__file__).with_name('test_packed_query.py')).read_text().split('rows=[]')[0])

if os.getenv('REAL_WEIGHTS')=='1':
    from safetensors import safe_open
    from sgl_kernel_npu.mla_prolog_v3_compat import dequant_mxfp8
    model=Path('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8')
    mapping=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    layer=int(os.getenv('LAYER','2'))
    prefix=f'model.layers.{layer}.self_attn.'
    def load(name):
        key=prefix+name
        with safe_open(str(model/mapping[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
    def projection(name):
        weight=dequant_mxfp8(load(name+'.weight'),load(name+'.weight_scale'))
        return nz(weight.t().contiguous().to(dev))
    wdq=projection('q_a_proj')
    wuq=projection('q_b_proj')
    wdkv=projection('kv_a_proj_with_mqa')
    wuk=load('kv_b_proj.weight').view(20,448,512)[:,:192].contiguous().to(dev)
    gq=load('q_a_layernorm.weight').to(dev)
    gkv=load('kv_a_layernorm.weight').to(dev)
    print('REAL CHECKPOINT',model,'layer',layer,'runtime BF16 mode0',flush=True)

def flags(head, parallel):
    os.environ['MLA_PROLOG_V3_DECODE_HEAD_LOCAL']=str(head)
    os.environ['MLA_PROLOG_V3_DECODE_PARALLEL_DOWN']=str(parallel)

rows=[]
sizes=[int(x) for x in os.getenv('SIZES','1,7,16,17,32,44,64,65,92').split(',')]
for t in sizes:
    args,kw,_=make(t,True)
    flags(0,0)
    os.environ["MLA_PROLOG_V3_DECODE_WIDE_N"]="0"
    os.environ['MLA_PROLOG_V3_CORES']='16'
    reference=C.mla_prolog_v3(*args,**kw)
    cache_ref=[args[9].clone(),args[10].clone()]
    variants=[('old',False,0,0,16,8,6),('packed',True,0,0,16,8,6),
        ('core20',True,0,0,20,8,6),('head20',True,1,0,20,8,6)]
    for cq,kv in [(8,6),(12,8),(10,9),(8,12),(8,9),(12,6)]:
        variants.append((f'parallel{cq}_{kv}',True,0,1,20,cq,kv))
        variants.append((f'both{cq}_{kv}',True,1,1,20,cq,kv))
    variants += [('wide16',True,0,0,16,8,6),('wide20',True,0,0,20,8,6),
        ('wide_head20',True,1,0,20,8,6),('wide_both10_9',True,1,1,20,10,9)]
    graphs=[]
    buffers=[]
    for label,packed,h,p,cores,cq,kv in variants:
        flags(h,p)
        os.environ["MLA_PROLOG_V3_DECODE_WIDE_N"]="1" if label.startswith("wide") else "0"
        os.environ['MLA_PROLOG_V3_CORES']=str(cores)
        os.environ['MLA_PROLOG_V3_DECODE_CQ_CORES']=str(cq)
        os.environ['MLA_PROLOG_V3_DECODE_KV_CORES']=str(kv)
        out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=packed)
        def run(out=out,packed=packed):
            C.mla_prolog_v3_out(*args,*out,**kw)
            return adapter.combine_query(out[0],out[1]) if packed else torch.cat([out[0],out[1]],-1)
        actual=run();torch.npu.synchronize()
        assert all(torch.equal(a,b) or a.numel()==b.numel()==0 for a,b in zip(reference,out)),(t,label,'outputs')
        assert torch.equal(cache_ref[0],args[9]) and torch.equal(cache_ref[1],args[10]),(t,label,'cache')
        graphs.append(capture(run));buffers.append(out)
    times=[[] for _ in variants]
    for rep in range(41):
        # Balanced cyclic/reverse order, covering each variant at every position.
        order=list(range(len(variants)))
        order=order[rep%len(order):]+order[:rep%len(order)]
        if rep%2:order.reverse()
        for j in order:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record();graphs[j].replay();end.record();end.synchronize()
            times[j].append(start.elapsed_time(end)*1000/REPEATS)
    latency={v[0]:statistics.median(tm) for v,tm in zip(variants,times)}
    row=dict(tokens=t,bit_exact=True,us=latency,
             speedups_vs_old={key:latency['old']/value for key,value in latency.items()},
             speedups_vs_packed={key:latency['packed']/value for key,value in latency.items()})
    rows.append(row);print(json.dumps(row),flush=True)

dest=Path(os.getenv('OUT','/tmp/mla-output576-20261030/ablation-results.json'))
dest.write_text(json.dumps(dict(rows=rows,seed=42,calls_per_graph=REPEATS,
                               timing_groups=41,core_env=os.getenv('MLA_PROLOG_V3_CORES'),
                               real_weights=os.getenv('REAL_WEIGHTS')=='1',layer=os.getenv('LAYER','2')),indent=2))
print('PASS',len(rows),'five-way decode ablations',flush=True)
