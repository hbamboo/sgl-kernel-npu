"""Two 16384-token prompts, total scheduled prefill budget 32768, on localhost."""
import json,os,time,urllib.request
from pathlib import Path
base='http://127.0.0.1:'+os.environ.get('PORT','19002')
for _ in range(120):
    try:
        ready=os.environ.get('READY_LOG')
        if ready and 'The server is fired up and ready to roll!' not in Path(ready).read_text():
            time.sleep(1);continue
        urllib.request.urlopen(base+'/model_info',timeout=2).read();break
    except Exception:time.sleep(1)
else:raise RuntimeError('server did not become ready')
prefix=int(os.environ.get('PREFIX_ID','32100'))
inputs=[[prefix]+[42]*16383,[prefix+1]+[43]*16383]
payload={'input_ids':inputs,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}}
req=urllib.request.Request(base+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
t0=time.perf_counter()
with urllib.request.urlopen(req,timeout=900) as r:data=json.load(r)
assert isinstance(data,list) and len(data)==2
assert all(r['meta_info']['cached_tokens']==0 for r in data), 'cache hit: choose fresh PREFIX_ID or restart server'
result={'seconds':time.perf_counter()-t0,'response':data}
path=os.environ.get('OUT','/tmp/mla_v3_batch32768.json')
with open(path,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
