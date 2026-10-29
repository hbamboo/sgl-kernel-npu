"""Teacher-forced comparison inputs from the same CHAT dataset as bench_glm.sh.
Run against one isolated server per variant; compare mean NLL over matching tokens.
This is a numerical stability check, not a substitute for model quality evaluation.
"""
import os,json,time,urllib.request,math
from pathlib import Path
from transformers import AutoTokenizer
base='http://127.0.0.1:'+os.environ.get('PORT','19002')
p=Path(os.environ['READY_LOG'])
for _ in range(180):
    if p.exists() and 'The server is fired up and ready to roll!' in p.read_text():break
    time.sleep(1)
else:raise RuntimeError('not ready')
tok=AutoTokenizer.from_pretrained('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8',trust_remote_code=True)
inputs=[];indices=[]
for i,line in enumerate(open('/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl')):
    if i%15:continue
    ids=tok.encode(json.loads(line)['conversations'][0]['content'])
    if 1024<=len(ids)<=16000:inputs.append(ids);indices.append(i)
    if len(inputs)==64:break
outputs=[]
for start in range(0,len(inputs),8):
    batch=inputs[start:start+8]
    payload={'input_ids':batch,'sampling_params':{'temperature':0,'max_new_tokens':1},'return_logprob':True,'logprob_start_len':[len(x)-256 for x in batch]}
    req=urllib.request.Request(base+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as response:out=json.load(response)
    for idx,ids,item in zip(indices[start:start+8],batch,out):
        meta=item['meta_info'];assert meta['cached_tokens']==0
        lp=[p[0] for p in meta['input_token_logprobs'] if p[0] is not None]
        outputs.append(dict(index=idx,length=len(ids),lp=lp,output_ids=item['output_ids']))
    print('finished',len(outputs),flush=True)
lp=[v for x in outputs for v in x['lp']]
result=dict(tokens=len(lp),nll=-sum(lp)/len(lp),ppl=math.exp(-sum(lp)/len(lp)),cases=outputs)
Path(os.environ['OUT']).write_text(json.dumps(result))
print({k:v for k,v in result.items() if k!='cases'},flush=True)
