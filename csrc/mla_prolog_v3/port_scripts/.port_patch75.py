"""Torch/PyBind layer: single thread-local cache for tiling tensor + workspace."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

# replace CachedTiling + WorkspaceFor with one thread-local entry keyed by the tiling key
OLD = '''// Tiling data buffer, cached per (tiling content, device) so that repeated launches
// (including ACL graph capture) reuse a stable device address.
at::Tensor &CachedTiling(const optiling::MlaPrologTilingData &tiling)
{
    static std::mutex mtx;
    static std::unordered_map<uint64_t, at::Tensor> cache;

    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    uint64_t key = HashBlob(&tiling, sizeof(tiling));
    key = HashBlob(&device_index, sizeof(device_index), key);

    std::lock_guard<std::mutex> guard(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto host = at::empty({static_cast<int64_t>(sizeof(tiling))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(host.data_ptr(), &tiling, sizeof(tiling));
    at::Tensor pinned = host.pin_memory();
    at::Tensor device = pinned.to(c10::Device(c10::DeviceType::PrivateUse1, device_index), pinned.scalar_type(),
                                  /*non_blocking=*/true, /*copy=*/true);
    return cache.emplace(key, std::move(device)).first->second;
}

// Scratch buffer on the current stream: one per-device allocation, grown on demand.
at::Tensor WorkspaceFor(int64_t bytes, const at::Tensor &ref)
{
    static std::mutex mtx;
    static std::unordered_map<int64_t, at::Tensor> cache;
    const int64_t dev_index = ref.device().index();
    std::lock_guard<std::mutex> guard(mtx);
    auto &slot = cache[dev_index];
    if (!slot.defined() || slot.numel() < bytes) {
        slot = at::empty({bytes}, at::TensorOptions().dtype(at::kByte).device(ref.device()));
    }
    return slot;
}'''

NEW = '''// One cache slot per tiling key: the device-side tiling blob plus the scratch buffer.
// Thread-local, so the hot path takes no lock and hashes nothing beyond the key the
// tiling cache already produced.  Serving threads are long lived, so the few duplicated
// entries per thread cost nothing.
struct LaunchSlot {
    at::Tensor tiling;
    at::Tensor workspace;
};

LaunchSlot &SlotFor(uint64_t key, int device_index, const optiling::MlaPrologTilingData &tiling, int64_t workspaceBytes,
                    const at::Tensor &ref)
{
    static thread_local std::unordered_map<uint64_t, LaunchSlot> cache;
    const uint64_t slotKey = key ^ (static_cast<uint64_t>(device_index) << 56);
    auto it = cache.find(slotKey);
    if (it != cache.end()) {
        LaunchSlot &slot = it->second;
        if (slot.workspace.numel() < workspaceBytes) {
            slot.workspace = at::empty({workspaceBytes}, at::TensorOptions().dtype(at::kByte).device(ref.device()));
        }
        return slot;
    }
    LaunchSlot slot;
    auto host = at::empty({static_cast<int64_t>(sizeof(tiling))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(host.data_ptr(), &tiling, sizeof(tiling));
    at::Tensor pinned = host.pin_memory();
    slot.tiling = pinned.to(c10::Device(c10::DeviceType::PrivateUse1, device_index), pinned.scalar_type(),
                            /*non_blocking=*/true, /*copy=*/true);
    slot.workspace = at::empty({workspaceBytes}, at::TensorOptions().dtype(at::kByte).device(ref.device()));
    return cache.emplace(slotKey, std::move(slot)).first->second;
}'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''    optiling::MlaPrologTilingData tilingData{};
    MlaPrologV3TilingResult res;
    std::string err;
    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err), "mla_prolog_v3 tiling failed: ", err);

    const at::Tensor &tiling_tensor = CachedTiling(tilingData);
    at::Tensor workspace = WorkspaceFor(static_cast<int64_t>(res.workspaceSize), token_x);'''
NEW2 = '''    optiling::MlaPrologTilingData tilingData{};
    MlaPrologV3TilingResult res;
    std::string err;
    uint64_t tilingKey = 0;
    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err, &tilingKey), "mla_prolog_v3 tiling failed: ",
                err);

    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    LaunchSlot &slot = SlotFor(tilingKey, device_index, tilingData, static_cast<int64_t>(res.workspaceSize), token_x);
    const at::Tensor &tiling_tensor = slot.tiling;
    at::Tensor &workspace = slot.workspace;'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("torch layer: single thread-local slot")
