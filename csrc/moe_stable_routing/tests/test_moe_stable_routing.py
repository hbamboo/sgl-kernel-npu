"""Manual installed-companion regression on an explicitly allocated idle card.

No server or framework patch. Compare all four native outputs before timing.
Timing always includes the entire three-kernel call, never just local Sort.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', type=Path, default=None)
    p.add_argument('--tokens', nargs='+', type=int, default=[8192, 8193, 12288, 16384, 16385, 32768, 36000])
    p.add_argument('--distributions', nargs='+', choices=['topk', 'all-one', 'repeated', 'boundary'],
                   default=['topk', 'all-one', 'repeated', 'boundary'])
    p.add_argument('--time', action='store_true', help='only with exclusive device/host performance window')
    a = p.parse_args()
    import torch
    import torch_npu
    import sgl_kernel_npu.moe_stable_routing as installed
    torch.set_num_threads(4)
    torch.npu.set_device(a.device)
    dev = torch.device(f'npu:{a.device}')
    formal = Path(installed.__file__).resolve().parent/'lib/libsgl_kernel_npu_moe_stable.so'
    if a.library is not None and a.library.resolve() != formal.resolve():
        p.error(f'--library must be the installed companion path: {formal}')
    a.library = formal
    report = {'status': 'running', 'device': str(dev), 'library': str(a.library.resolve()),
              'library_sha256': hashlib.sha256(a.library.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'cases': [], 'timing_enabled': a.time,
              'installed_module': str(Path(installed.__file__).resolve()),
              'installed_module_sha256': hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest(),
              'timing_scope': 'prequantized native versus entire 3-kernel call; same prior quantization excluded from both'}
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = a.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(a.output)

    def compare(expected, actual):
        fields = []
        for name, lhs, rhs in zip(('payload', 'inverse', 'counts', 'scale'), expected, actual):
            # The current framework normalizes native scale[N,64] to [N,32,2].
            if name == 'scale':
                lhs = lhs.reshape(-1, 32, 2)
            lb = lhs.contiguous().view(torch.uint8).cpu()
            rb = rhs.contiguous().view(torch.uint8).cpu()
            same = lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and torch.equal(lb, rb)
            fields.append({'field': name, 'shape_expected': list(lhs.shape), 'shape_actual': list(rhs.shape),
                           'dtype_expected': str(lhs.dtype), 'dtype_actual': str(rhs.dtype),
                           'bytes': lb.numel(), 'exact': same,
                           'different_bytes': int((lb != rb).sum()) if lb.shape == rb.shape else None})
            if not same:
                raise AssertionError(json.dumps(fields[-1]))
        return fields

    def measure(fn):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        device_us, wall_us = [], []
        for _ in range(21):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            start.record(); outputs = fn(); end.record()
            end.synchronize()
            wall_us.append((time.perf_counter()-t0)*1e6)
            device_us.append(start.elapsed_time(end)*1000)
            del outputs
        return {'device_event_us': device_us, 'device_median_us': statistics.median(device_us),
                'host_completed_us': wall_us, 'host_completed_median_us': statistics.median(wall_us)}

    save()
    try:
        for tokens in a.tokens:
            generator = torch.Generator().manual_seed(1031+tokens)
            source = torch.randn(tokens, 2048, dtype=torch.bfloat16, generator=generator).to(dev)
            x, scale = torch.ops.npu.npu_dynamic_mx_quant(source, axis=-1, round_mode='rint',
                    dst_type=torch.float8_e4m3fn, block_size=32, scale_alg=0)
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e8m0fnu)
            for distribution in a.distributions:
                if distribution == 'topk':
                    ids = torch.rand(tokens,64,generator=generator).topk(4,dim=-1).indices.to(torch.int32)
                elif distribution == 'all-one':
                    ids = torch.full((tokens,4),63,dtype=torch.int32)
                elif distribution == 'repeated':
                    ids = ((torch.arange(tokens,dtype=torch.int32)//37)%64).unsqueeze(1).expand(-1,4).contiguous()
                else:
                    ids = ((torch.arange(tokens*4,dtype=torch.int32)*17+
                            torch.arange(tokens*4,dtype=torch.int32)//4*7)%64).reshape(tokens,4)
                ids = ids.to(dev)
                kwargs = {'active_num': tokens*4, 'expert_num': 64, 'expert_tokens_num_type': 1,
                          'expert_tokens_num_flag': True, 'active_expert_range': [0,64], 'quant_mode': -1}
                native = lambda: torch.ops.npu.npu_moe_init_routing_v2(x, ids, scale=scale, **kwargs)
                def candidate():
                    output = installed.route_prequantized(x, ids, scale)
                    if output is None:
                        raise RuntimeError('installed companion unavailable; native fallback is not a candidate test')
                    return output
                expected = native(); actual = candidate(); torch.npu.synchronize()
                case = {'tokens': tokens, 'distribution': distribution, 'status': 'checking'}
                report['cases'].append(case); save()
                case['outputs'] = compare(expected, actual)
                hist = torch.bincount(ids.cpu().reshape(-1).long(), minlength=64)
                assert torch.equal(actual[2].cpu(), hist)
                mapping = actual[1].cpu().long()
                assert torch.equal(mapping.sort().values, torch.arange(tokens*4))
                case['histogram_and_inverse_bijection'] = True
                # Nondefault caller stream and returned-workspace allocator lifetime.
                stream = torch.npu.Stream(); stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    alternate = candidate()
                    consume = alternate[0].view(torch.uint8).clone()
                stream.synchronize()
                compare(expected, alternate)
                assert torch.equal(consume.cpu(), alternate[0].view(torch.uint8).cpu())
                case['nondefault_stream_exact'] = True
                if tokens == 8192 and distribution == 'topk':
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        graph_output = candidate()
                    for factor in (0.25,-0.75):
                        changed = source*factor
                        q2,s2 = torch.ops.npu.npu_dynamic_mx_quant(changed, axis=-1, round_mode='rint',
                            dst_type=torch.float8_e4m3fn,block_size=32,scale_alg=0)
                        if s2.dtype == torch.uint8:
                            s2 = s2.view(torch.float8_e8m0fnu)
                        x.copy_(q2); scale.copy_(s2)
                        ids.copy_(torch.rand(tokens,64,device=dev).topk(4,dim=-1).indices.to(torch.int32))
                        graph_expected = native()
                        graph.replay(); torch.npu.synchronize()
                        compare(graph_expected, graph_output)
                    case['graph_replays_with_new_x_ids_scales'] = 2
                    del graph, graph_output, graph_expected, changed, q2, s2
                if a.time:
                    # Alternate native/candidate blocks; retain every sample.
                    case['timing'] = [{'arm': name, **measure(fn)} for name,fn in
                                      [('native',native), ('candidate',candidate), ('candidate',candidate), ('native',native)]]
                case['status'] = 'passed'; save()
                print(json.dumps(case), flush=True)
                del expected, actual, alternate, consume
        report['status'] = 'passed'; save()
    except BaseException as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'; save()
        raise


if __name__ == '__main__':
    main()
