#!/usr/bin/env python3
"""Add a tiling cache + reusable workspace to the MlaPrologV3 torch entry.

Running the ops-transformer tiling costs ~40 us per call (matmul tiling + K/N
search), which dominates the 16.5 MiB of weights the kernel has to stream.
Tiling only depends on shapes/dtypes/attrs, so cache it.  The workspace is pure
scratch memory on the current stream and can be reused as well.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.h")
C = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp")
T = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

# ---------------------------------------------------------------- header
with io.open(H, encoding="utf-8") as f:
    s = f.read()
old = """bool MlaPrologV3RunTiling(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                          MlaPrologV3TilingResult *result, std::string *err);"""
new = """bool MlaPrologV3RunTiling(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                          MlaPrologV3TilingResult *result, std::string *err);

// Same as above but memoised on the full (shape, dtype, attr) configuration:
// the tiling costs tens of microseconds and only depends on those inputs.
bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err);"""
assert s.count(old) == 1
with io.open(H, "w", encoding="utf-8") as f:
    f.write(s.replace(old, new))

# ---------------------------------------------------------------- impl
CACHE_IMPL = r"""
namespace {

void AppendDims(std::string *key, const std::vector<int64_t> &dims)
{
    key->append(std::to_string(dims.size()));
    for (int64_t d : dims) {
        key->append(",");
        key->append(std::to_string(d));
    }
    key->append(";");
}

void AppendAttr(std::string *key, double v)
{
    key->append(std::to_string(v));
    key->append(";");
}

std::string MakeTilingKey(const MlaPrologV3Tensors &t)
{
    std::string key;
    key.reserve(1024);
    AppendDims(&key, t.tokenX);
    AppendDims(&key, t.weightDq);
    AppendDims(&key, t.weightUqQr);
    AppendDims(&key, t.weightUk);
    AppendDims(&key, t.weightDkvKr);
    AppendDims(&key, t.gammaCq);
    AppendDims(&key, t.gammaCkv);
    AppendDims(&key, t.ropeSin);
    AppendDims(&key, t.ropeCos);
    AppendDims(&key, t.kvCache);
    AppendDims(&key, t.krCache);
    AppendDims(&key, t.cacheIndex);
    AppendDims(&key, t.dequantScaleX);
    AppendDims(&key, t.dequantScaleWDq);
    AppendDims(&key, t.dequantScaleWUqQr);
    AppendDims(&key, t.dequantScaleWDkvKr);
    AppendDims(&key, t.quantScaleCkv);
    AppendDims(&key, t.quantScaleCkr);
    AppendDims(&key, t.smoothScalesCq);
    AppendDims(&key, t.actualSeqLen);
    AppendDims(&key, t.kNopeClipAlpha);
    AppendDims(&key, t.query);
    AppendDims(&key, t.queryRope);
    AppendDims(&key, t.queryNorm);
    AppendDims(&key, t.dequantScaleQNorm);
    const int dtypes[] = {static_cast<int>(t.tokenXType),   static_cast<int>(t.weightDqType),
                          static_cast<int>(t.weightUqQrType), static_cast<int>(t.weightUkType),
                          static_cast<int>(t.weightDkvKrType), static_cast<int>(t.gammaCqType),
                          static_cast<int>(t.gammaCkvType), static_cast<int>(t.ropeSinType),
                          static_cast<int>(t.ropeCosType), static_cast<int>(t.kvCacheType),
                          static_cast<int>(t.krCacheType), static_cast<int>(t.scaleXType),
                          static_cast<int>(t.scaleWDqType), static_cast<int>(t.scaleWUqQrType),
                          static_cast<int>(t.scaleWDkvKrType), static_cast<int>(t.scaleQNopeType),
                          static_cast<int>(t.scaleQNormType), static_cast<int>(t.queryNormType)};
    for (int d : dtypes) {
        key.append(std::to_string(d));
        key.append(",");
    }
    key.append(t.cacheMode);
    key.append(";");
    AppendAttr(&key, t.rmsEpsCq);
    AppendAttr(&key, t.rmsEpsCkv);
    AppendAttr(&key, t.weightQuantMode);
    AppendAttr(&key, t.kvQuantMode);
    AppendAttr(&key, t.queryQuantMode);
    AppendAttr(&key, t.ckvkrRepoMode);
    AppendAttr(&key, t.quantScaleRepoMode);
    AppendAttr(&key, t.tileSize);
    AppendAttr(&key, t.qcQrScale);
    AppendAttr(&key, t.kcScale);
    AppendAttr(&key, t.queryNormFlag);
    AppendAttr(&key, t.doRope);
    return key;
}

} // namespace

bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err)
{
    struct Entry {
        optiling::MlaPrologTilingData tilingData;
        MlaPrologV3TilingResult result;
        bool ok;
        std::string err;
    };
    static std::mutex mutex;
    static std::unordered_map<std::string, Entry> cache;

    const std::string key = MakeTilingKey(tensors);
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            *tilingData = it->second.tilingData;
            *result = it->second.result;
            if (!it->second.ok && err != nullptr) {
                *err = it->second.err;
            }
            return it->second.ok;
        }
    }

    Entry entry{};
    entry.ok = MlaPrologV3RunTiling(tensors, &entry.tilingData, &entry.result, &entry.err);

    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.emplace(key, entry).first;
    *tilingData = it->second.tilingData;
    *result = it->second.result;
    if (!it->second.ok && err != nullptr) {
        *err = it->second.err;
    }
    return it->second.ok;
}
"""

with io.open(C, encoding="utf-8") as f:
    s = f.read()
# needs <mutex>/<unordered_map> and <string>
s = s.replace('#include <cstring>\n#include <deque>', '#include <cstring>\n#include <deque>\n#include <mutex>\n#include <unordered_map>')
marker = """} // namespace npu_kernel
} // namespace sglang"""
assert s.count(marker) == 1
s = s.replace(marker, CACHE_IMPL + "\n" + marker)
with io.open(C, "w", encoding="utf-8") as f:
    f.write(s)

# ---------------------------------------------------------------- torch entry
with io.open(T, encoding="utf-8") as f:
    s = f.read()
pairs = [
    ("""    TORCH_CHECK(MlaPrologV3RunTiling(t, &tilingData, &res, &err), "mla_prolog_v3 tiling failed: ", err);""",
     """    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err), "mla_prolog_v3 tiling failed: ", err);"""),
    ("""    const at::Tensor &tiling_tensor = CachedTiling(tilingData);
    at::Tensor workspace =
        at::empty({static_cast<int64_t>(res.workspaceSize)}, at::TensorOptions().dtype(at::kByte).device(token_x.device()));""",
     """    const at::Tensor &tiling_tensor = CachedTiling(tilingData);
    // Scratch buffer on the current stream: reuse a per-device allocation instead
    // of asking the caching allocator on every launch.
    static std::mutex ws_mutex;
    static std::unordered_map<int64_t, at::Tensor> ws_cache;
    at::Tensor workspace;
    {
        const int64_t bytes = static_cast<int64_t>(res.workspaceSize);
        const int64_t dev_index = token_x.device().index();
        std::lock_guard<std::mutex> guard(ws_mutex);
        auto &slot = ws_cache[dev_index];
        if (!slot.defined() || slot.numel() < bytes) {
            slot = at::empty({bytes}, at::TensorOptions().dtype(at::kByte).device(token_x.device()));
        }
        workspace = slot;
    }"""),
]
for old, new in pairs:
    n = s.count(old)
    if n != 1:
        raise SystemExit("pattern count %d for %r" % (n, old[:60]))
    s = s.replace(old, new)
with io.open(T, "w", encoding="utf-8") as f:
    f.write(s)
print("added tiling cache + workspace reuse")
