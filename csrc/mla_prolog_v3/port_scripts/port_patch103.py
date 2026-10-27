"""Route A-lite: kernel side of the concurrent mm1/mm2 schedule.

mm2 occupies block range [mm2BlockStart, mm2BlockStart + mm2BlockNum), so its offsets must be
relative to that base and its MatmulSplitN must be restricted to the range.  mm1 keeps
[0, mm1BlockNum) and its offsets are unchanged (mm2BlockStart is 0 unless the env flag is on).
"""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h"
s = io.open(P, encoding="utf-8").read()

if "mm2BlockIdx" not in s:
    old = ("    aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_;"
           " //  (512 + 64) / 9 * idx  = 64 * idx")
    assert s.count(old) == 1, s.count(old)
    new = ("    // [port] mm2 may own a disjoint block range above mm1's (concurrent schedule)\n"
           "    const int64_t mm2BlockIdx = static_cast<int64_t>(blockIdx_) -"
           " static_cast<int64_t>(baseParams_->mm2BlockStart);\n"
           "    aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * mm2BlockIdx;"
           " //  (512 + 64) / 9 * idx  = 64 * idx")
    s = s.replace(old, new)

    old2 = ("        mmCkvKrParam_, UsedBlockParams{0, baseParams_->mm2BlockNum}, scaleAGm, scaleBGmDkvKr);")
    assert s.count(old2) == 1, s.count(old2)
    new2 = ("        mmCkvKrParam_, UsedBlockParams{baseParams_->mm2BlockStart,"
            " baseParams_->mm2BlockStart + baseParams_->mm2BlockNum}, scaleAGm, scaleBGmDkvKr);")
    s = s.replace(old2, new2)
    io.open(P, "w", encoding="utf-8").write(s)
    print("kernel: concurrent mm2 range wired")
else:
    print("kernel: already patched")
