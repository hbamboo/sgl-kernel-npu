"""Original16-core+cat / packed-only / automatic decode, with real weights."""
import os,json,statistics,time
from pathlib import Path

# Load the validated random-input generator and actual layer2 checkpoint weights.
os.environ['REAL_WEIGHTS']='1'
exec(Path(__file__).with_name('ablate_decode.py').read_text().split('def flags(')[0])

def select(label):
    for name in ['CORES','DECODE_HEAD_LOCAL','DECODE_PARALLEL_DOWN','DECODE_CQ_CORES','DECODE_KV_CORES']:
        os.environ.pop('MLA_PROLOG_V3_'+name,None)
    os.environ['MLA_PROLOG_V3_DECODE_TUNE']='1' if label=='automatic' else '0'

sizes=[int(x) for x in os.getenv('SIZES','1,7,16,17,32,33,40,41,44,48,64,65,80,92,127,128,129').split(',')]
rows=[]
for t in sizes:
    args,kw,_=make(t,True)
    select('old')
    reference=C.mla_prolog_v3(*args,**kw)
    cache_ref=[args[9].clone(),args[10].clone()]
    graphs=[];buffers=[]
    variants=['old','packed','automatic']
    for label in variants:
        select(label)
        packed=label!='old'
        out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=packed)
        def run(out=out,packed=packed):
            C.mla_prolog_v3_out(*args,*out,**kw)
            return adapter.combine_query(out[0],out[1]) if packed else torch.cat([out[0],out[1]],-1)
        run();torch.npu.synchronize()
        assert all(torch.equal(a,b) or a.numel()==b.numel()==0 for a,b in zip(reference,out)),(t,label)
        assert torch.equal(cache_ref[0],args[9]) and torch.equal(cache_ref[1],args[10]),(t,label,'cache')
        graphs.append(capture(run));buffers.append(out)
    times=[[] for _ in variants];host_times=[[] for _ in variants]
    for rep in range(41):
        order=[0,1,2] if rep%2==0 else [2,1,0]
        for j in order:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            before=time.perf_counter_ns()
            start.record();graphs[j].replay();end.record()
            host_times[j].append((time.perf_counter_ns()-before)/1000)
            end.synchronize()
            times[j].append(start.elapsed_time(end)*1000/REPEATS)
    us={label:statistics.median(tm) for label,tm in zip(variants,times)}
    row=dict(tokens=t,bit_exact=True,us=us,speedup=us['old']/us['automatic'],
             latency_reduction_pct=(1-us['automatic']/us['old'])*100,
             host_enqueue_us=[statistics.median(tm) for tm in host_times])
    rows.append(row);print(json.dumps(row),flush=True)
dest=Path(os.getenv('OUT','/tmp/mla-output576-20261030/final-comparison.json'))
dest.write_text(json.dumps(dict(rows=rows,real_checkpoint=str(model),layer=layer,
                               calls_per_graph=REPEATS,pairs=41),indent=2))
print('PASS final decode comparison',len(rows),flush=True)
