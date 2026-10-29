import csv,json,statistics,sys
from pathlib import Path
from decimal import Decimal as D
p=Path(sys.argv[1]);events=json.load(open(next(p.rglob('trace_view.json'))))
ranges=[e for e in events if e.get('name')=='mla_prolog_v3_prefill_bf16' and e.get('ph')=='X']
rows=list(csv.DictReader(open(next(p.rglob('kernel_details.csv')))))
kernels={D(r['Start Time(us)'].strip()):r for r in rows if r['Accelerator Core'] in ('AI_CORE','AI_VECTOR_CORE','AI_MIX_CORE')}
starts={e['id']:e for e in events if e.get('cat')=='async_npu' and e.get('ph')=='s'}
finishes={e['id']:e for e in events if e.get('cat')=='async_npu' and e.get('ph')=='f'}
links=[(D(a['ts']),D(finishes[i]['ts'])) for i,a in starts.items() if i in finishes]
result=[]
for rg in ranges:
    begin=D(rg['ts']);end=begin+D(str(rg['dur']))
    ks=[kernels[b] for a,b in links if begin<=a<end and b in kernels]
    if not ks:continue
    s=[D(k['Start Time(us)'].strip()) for k in ks];ends=[a+D(k['Duration(us)']) for a,k in zip(s,ks)]
    result.append(dict(kernels=len(ks),sum_core_us=sum(float(k['Duration(us)']) for k in ks),span_us=float(max(ends)-min(s))))
print('ranges',len(ranges),'matched',len(result))
for key in ('kernels','sum_core_us','span_us'):
    v=[r[key] for r in result];print(key,'median',statistics.median(v),'min',min(v),'max',max(v))
(p/'mla_scope_summary.json').write_text(json.dumps(result,indent=2))
