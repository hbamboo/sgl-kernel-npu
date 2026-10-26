"""Host key exposure (matching the actual code) + cached empty tensor in the torch layer."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P1 = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD1 = """bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err);"""
NEW1 = """bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err,
                                uint64_t *outKey = nullptr);"""
if s.count(OLD1) == 1:
    s = s.replace(OLD1, NEW1)
    with io.open(P1, "w", encoding="utf-8") as f:
        f.write(s)
    print("header updated")

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err)
{"""
NEW2 = """bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err, uint64_t *outKey)
{"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = """    const uint64_t key = MakeTilingHash(tensors);
    {"""
NEW3 = """    const uint64_t key = MakeTilingHash(tensors);
    if (outKey != nullptr) {
        *outKey = key;
    }
    {"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("host updated")
