"""A/B: FlashMLA vs FIA prefill, same BF16 prolog and FlashMLA decode, card 7.

OUT sets the parent folder. NUM_PROMPTS=2000 and REQUEST_RATE=26 by default.
Every run uses a new timestamped folder; only child servers are terminated.
"""
import os,time,json,socket,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[4]
out=Path(os.environ.get('OUT','/tmp/mla_prolog_v3_prefill_attention_ab'))/time.strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True,exist_ok=False)
num=int(os.environ.get('NUM_PROMPTS','2000'));rate=os.environ.get('REQUEST_RATE','26');port=os.environ.get('PORT','19002')
print('output_dir',out,flush=True)
def open_port():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1',int(port)))==0
summary=dict(num_prompts=num,request_rate=rate,device=7,prolog='BF16 prepared v3',decode='FlashMLA',variants={})
for name,flag in [('flashmla','1'),('fia','0')]:
    if open_port():raise RuntimeError(f'Port {port} is already in use')
    env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='7',PORT=port,
        ENABLE_ASCEND_FLASH_MLA_PREFILL=flag,SGLANG_NPU_USE_MLAPO_V3_PREFILL='1',
        SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE='1',SGLANG_NPU_MLAPO_V3_DEBUG='1')
    env.pop('SGLANG_NPU_MLAPO_V3_CAPTURE_DIR',None)
    slog=out/f'server_{name}.log'
    with slog.open('w') as sf:
        server=subprocess.Popen(['bash',str(root/'csrc/mla_prolog_v3/tests/e2e/run_prefill_7.sh'),'--disable-radix-cache'],cwd=root,env=env,stdout=sf,stderr=subprocess.STDOUT)
        (out/'active_server.json').write_text(json.dumps(dict(variant=name,pid=server.pid)))
        try:
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError(f'{name} server exited; see {slog}')
                if 'The server is fired up and ready to roll!' in slog.read_text():break
                time.sleep(1)
            else:raise RuntimeError(f'{name} server did not become ready')
            print('quality_start',name,'server_pid',server.pid,flush=True)
            env.update(READY_LOG=str(slog),OUT=str(out/f'quality_{name}.json'))
            with (out/f'quality_{name}.log').open('w') as qf:
                subprocess.run(['python3',str(root/'csrc/mla_prolog_v3/tests/e2e/check_prefill_logprobs.py')],cwd=root,env=env,stdout=qf,stderr=subprocess.STDOUT,check=True,timeout=900)
            print('benchmark_start',name,flush=True)
            cmd=['python3','bench_rate_npu_sv_genedata.py','--backend','sglang','--model','/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8','--port',port,'--log-dir',str(out/f'bench_{name}'),'--dataset-path','/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl','--rates',rate,'--npu-ids','7+0','--num-prompts',str(num)]
            with (out/f'bench_{name}.log').open('w') as bf:
                subprocess.run(cmd,cwd='/mnt/sfs_turbo/hxd/script',env=env,stdout=bf,stderr=subprocess.STDOUT,check=True,timeout=1800)
            paths=list(out.glob(f'bench_{name}_*/bench_output_rate*.jsonl'));assert len(paths)==1,paths
            result=json.loads(paths[0].read_text().splitlines()[-1]);assert result['completed']==num,result
            summary['variants'][name]={k:result[k] for k in ('completed','duration','total_input_tokens','total_output_tokens','request_throughput','output_throughput','total_throughput','mean_ttft_ms','mean_tpot_ms')}
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('benchmark_done',name,json.dumps(summary['variants'][name]),flush=True)
        finally:
            if server.poll() is None:server.terminate()
            server.wait(timeout=60)
            for _ in range(30):
                if not open_port():break
                time.sleep(1)
a=summary['variants']['flashmla'];b=summary['variants']['fia']
assert a['total_input_tokens']==b['total_input_tokens'] and a['total_output_tokens']==b['total_output_tokens']
qa=json.loads((out/'quality_flashmla.json').read_text());qb=json.loads((out/'quality_fia.json').read_text())
assert len(qa['cases'])==len(qb['cases'])
for x,y in zip(qa['cases'],qb['cases']):
    assert x['index']==y['index'] and x['length']==y['length'] and len(x['lp'])==len(y['lp'])
summary['quality']={
    'cases':len(qa['cases']),'tokens':qa['tokens'],
    'flashmla_nll':qa['nll'],'fia_nll':qb['nll'],
    'delta_nll':qb['nll']-qa['nll'],
    'ppl_relative_change':qb['ppl']/qa['ppl']-1,
    'next_token_matches':sum(x['output_ids']==y['output_ids'] for x,y in zip(qa['cases'],qb['cases']))}
summary['fia_throughput_relative_change']=b['total_throughput']/a['total_throughput']-1
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print('ALL_DONE',json.dumps(summary),flush=True)
