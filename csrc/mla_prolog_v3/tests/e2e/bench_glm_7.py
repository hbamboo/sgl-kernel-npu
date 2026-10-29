"""Run raw and prepared BF16 prefill with bench_glm.sh client, on card 7 only.
OUT sets the parent output folder; a timestamped child preserves previous runs.
The runner starts one server at a time and terminates only its own server.
"""
import os,time,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[4]
out=Path(os.environ.get('OUT','/tmp/mla_prolog_v3_full2000'))/time.strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True,exist_ok=False)
for name,flag in [('base','0'),('candidate','1')]:
    slog=out/f'server_full2000_{name}.log';blog=out/f'bench_full2000_{name}.log'
    env=os.environ.copy();env['SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE']=flag
    with slog.open('w') as sf:
        server=subprocess.Popen(['bash',str(root/'csrc/mla_prolog_v3/tests/e2e/run_prefill_7.sh'),'--disable-radix-cache'],cwd=root,env=env,stdout=sf,stderr=subprocess.STDOUT)
        (out/'full2000_server_pid.json').write_text(json.dumps(dict(variant=name,pid=server.pid)))
        try:
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError(f'{name} server exited')
                if 'The server is fired up and ready to roll!' in slog.read_text():break
                time.sleep(1)
            else:raise RuntimeError('server not ready')
            cmd=['python3','bench_rate_npu_sv_genedata.py','--backend','sglang','--model','/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8','--port','19002','--log-dir',str(out/f'bench_full2000_{name}'),'--dataset-path','/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl','--rates','26','--npu-ids','7+0','--num-prompts','2000']
            print('benchmark_start',name,'server_pid',server.pid,flush=True)
            with blog.open('w') as bf:subprocess.run(cmd,cwd='/mnt/sfs_turbo/hxd/script',env=env,stdout=bf,stderr=subprocess.STDOUT,check=True,timeout=1800)
            print('benchmark_done',name,flush=True)
        finally:
            if server.poll() is None:server.terminate()
            server.wait(timeout=60)
print('all_done',flush=True)
