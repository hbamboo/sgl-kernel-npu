#!/usr/bin/env python3
"""Replace the string-based cache keys with allocation-free 64-bit hashes.

The tiling and tiling-data caches ran on every launch; building the string keys
(std::to_string + append ~40 times) was a measurable part of the ~27 us host
cost at decode.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp")
T = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(H, encoding="utf-8") as f:
    s = f.read()

old_key = s[s.index("void AppendDims(std::string *key"):s.index("} // namespace\n\nbool MlaPrologV3RunTilingCached")]
new_key = r"""// Allocation-free FNV-1a over the full (shape, dtype, attr) configuration.
inline void HashBytes(uint64_t *h, const void *data, size_t bytes)
{
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        *h ^= static_cast<uint64_t>(p[i]);
        *h *= 1099511628211ULL;
    }
}

inline void HashDims(uint64_t *h, const std::vector<int64_t> &dims)
{
    const int64_t n = static_cast<int64_t>(dims.size());
    HashBytes(h, &n, sizeof(n));
    if (!dims.empty()) {
        HashBytes(h, dims.data(), dims.size() * sizeof(int64_t));
    }
}

inline void HashValue(uint64_t *h, int64_t v)
{
    HashBytes(h, &v, sizeof(v));
}

uint64_t MakeTilingHash(const MlaPrologV3Tensors &t)
{
    uint64_t h = 1469598103934665603ULL;
    HashDims(&h, t.tokenX);
    HashDims(&h, t.weightDq);
    HashDims(&h, t.weightUqQr);
    HashDims(&h, t.weightUk);
    HashDims(&h, t.weightDkvKr);
    HashDims(&h, t.gammaCq);
    HashDims(&h, t.gammaCkv);
    HashDims(&h, t.ropeSin);
    HashDims(&h, t.ropeCos);
    HashDims(&h, t.kvCache);
    HashDims(&h, t.krCache);
    HashDims(&h, t.cacheIndex);
    HashDims(&h, t.dequantScaleX);
    HashDims(&h, t.dequantScaleWDq);
    HashDims(&h, t.dequantScaleWUqQr);
    HashDims(&h, t.dequantScaleWDkvKr);
    HashDims(&h, t.quantScaleCkv);
    HashDims(&h, t.quantScaleCkr);
    HashDims(&h, t.smoothScalesCq);
    HashDims(&h, t.actualSeqLen);
    HashDims(&h, t.kNopeClipAlpha);
    HashDims(&h, t.query);
    HashDims(&h, t.queryRope);
    HashDims(&h, t.queryNorm);
    HashDims(&h, t.dequantScaleQNorm);
    const int dtypes[] = {static_cast<int>(t.tokenXType),      static_cast<int>(t.weightDqType),
                          static_cast<int>(t.weightUqQrType),  static_cast<int>(t.weightUkType),
                          static_cast<int>(t.weightDkvKrType), static_cast<int>(t.gammaCqType),
                          static_cast<int>(t.gammaCkvType),    static_cast<int>(t.ropeSinType),
                          static_cast<int>(t.ropeCosType),     static_cast<int>(t.kvCacheType),
                          static_cast<int>(t.krCacheType),     static_cast<int>(t.scaleXType),
                          static_cast<int>(t.scaleWDqType),    static_cast<int>(t.scaleWUqQrType),
                          static_cast<int>(t.scaleWDkvKrType), static_cast<int>(t.scaleQNopeType),
                          static_cast<int>(t.scaleQNormType),  static_cast<int>(t.queryNormType)};
    HashBytes(&h, dtypes, sizeof(dtypes));
    HashBytes(&h, t.cacheMode.data(), t.cacheMode.size());
    const double attrs[] = {t.rmsEpsCq,     t.rmsEpsCkv,      static_cast<double>(t.weightQuantMode),
                            static_cast<double>(t.kvQuantMode), static_cast<double>(t.queryQuantMode),
                            static_cast<double>(t.ckvkrRepoMode), static_cast<double>(t.quantScaleRepoMode),
                            static_cast<double>(t.tileSize), static_cast<double>(t.qcQrScale),
                            static_cast<double>(t.kcScale), static_cast<double>(t.queryNormFlag),
                            static_cast<double>(t.doRope)};
    HashBytes(&h, attrs, sizeof(attrs));
    return h;
}

"""
s = s.replace(old_key + "} // namespace\n\n", new_key + "} // namespace\n\n")

s = s.replace("""    static std::mutex mutex;
    static std::unordered_map<std::string, Entry> cache;

    const std::string key = MakeTilingKey(tensors);""",
              """    static std::mutex mutex;
    static std::unordered_map<uint64_t, Entry> cache;

    const uint64_t key = MakeTilingHash(tensors);""")
with io.open(H, "w", encoding="utf-8") as f:
    f.write(s)

# ---------------------------------------------------------------- tiling tensor cache
with io.open(T, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    std::string key(reinterpret_cast<const char *>(&tiling), sizeof(tiling));
    key.append(reinterpret_cast<const char *>(&device_index), sizeof(device_index));""",
              """    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    // allocation-free hash of the tiling blob + device
    uint64_t key = 1469598103934665603ULL;
    {
        const auto *p = reinterpret_cast<const unsigned char *>(&tiling);
        for (size_t i = 0; i < sizeof(tiling); ++i) {
            key ^= static_cast<uint64_t>(p[i]);
            key *= 1099511628211ULL;
        }
        const auto *d = reinterpret_cast<const unsigned char *>(&device_index);
        for (size_t i = 0; i < sizeof(device_index); ++i) {
            key ^= static_cast<uint64_t>(d[i]);
            key *= 1099511628211ULL;
        }
    }""")
s = s.replace("""    static std::mutex mtx;
    static std::unordered_map<std::string, at::Tensor> cache;""",
              """    static std::mutex mtx;
    static std::unordered_map<uint64_t, at::Tensor> cache;""")
with io.open(T, "w", encoding="utf-8") as f:
    f.write(s)
print("hash keys in place")
