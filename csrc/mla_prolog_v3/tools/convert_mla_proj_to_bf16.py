"""Build a bf16 overlay of the GLM-4.7-Flash MXFP8 checkpoint's MLA projections.

SGLang's MLAPO path (``forward_mlaprolog``) assumes the W8A8-int8 modelslim scheme:
it transposes ``qkv_a_proj.weight`` expecting ``[out, in]``, reads
``q_b_proj.weight_scale`` (deleted by the MXFP8 scheme) and forwards no scales for the
two down projections.  With an MXFP8 checkpoint it therefore cannot work as-is.

This tool writes the three MLA projections of every layer as bf16 into one extra
safetensors file, points the original index at it, and marks those keys FLOAT in the
quant description (dropping their block scales).  All other weights stay in the
original shards, which are symlinked, so the overlay costs ~700 MB instead of 30 GB.

    python convert_mla_proj_to_bf16.py <src-model-dir> <dst-dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK = 32
PROJ = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa")
OVERLAY = "mla_proj_bf16.safetensors"


def e8m0(u8):
    e = u8.to(torch.int32)
    val = torch.pow(2.0, (e - 127).float())
    return torch.where(e == 0, torch.zeros_like(val), val)


def dequant(w_fp8, scale_u8):
    w = w_fp8.to(torch.float32)
    s = e8m0(scale_u8).repeat_interleave(BLOCK, dim=1)[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "quant_model_weights.safetensors.index.json")))
    weight_map = dict(index["weight_map"])
    desc = json.load(open(os.path.join(src, "quant_model_description.json")))

    layers = sorted({int(k.split(".")[2]) for k in weight_map if k.startswith("model.layers.")})
    print("layers:", len(layers), flush=True)

    tensors = {}
    for layer in layers:
        p = "model.layers.%d.self_attn." % layer
        for proj in PROJ:
            wkey, skey = p + proj + ".weight", p + proj + ".weight_scale"
            if skey not in weight_map:
                continue
            with safe_open(os.path.join(src, weight_map[wkey]), framework="pt", device="cpu") as f:
                tensors[wkey] = dequant(f.get_tensor(wkey), f.get_tensor(skey))
            weight_map[wkey] = OVERLAY
            # the scale key must go away in BOTH the index and the description,
            # otherwise the loader still treats the layer as quantised
            weight_map.pop(skey, None)
            desc.pop(skey, None)
            desc[wkey] = "FLOAT"
        if (layer + 1) % 10 == 0:
            print("  converted %d/%d layers" % (layer + 1, len(layers)), flush=True)

    save_file(tensors, os.path.join(dst, OVERLAY), metadata={"format": "pt"})
    print("overlay: %.1f MiB, %d tensors"
          % (os.path.getsize(os.path.join(dst, OVERLAY)) / 2**20, len(tensors)), flush=True)

    new_index = {"metadata": index.get("metadata", {}), "weight_map": weight_map}
    json.dump(new_index, open(os.path.join(dst, "quant_model_weights.safetensors.index.json"), "w"), indent=2)
    json.dump(desc, open(os.path.join(dst, "quant_model_description.json"), "w"), indent=2)

    for name in sorted({v for v in index["weight_map"].values()}):
        target = os.path.join(dst, name)
        if not os.path.exists(target):
            os.symlink(os.path.join(src, name), target)
    for name in os.listdir(src):
        s_path, d_path = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d_path):
            continue
        if name.endswith(".safetensors"):
            continue
        if os.path.isdir(s_path):
            continue
        os.symlink(s_path, d_path)
    print("done ->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
