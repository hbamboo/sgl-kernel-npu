#include "scatter_pa_compression_entry.h"
extern "C" __global__ __aicore__ void sgl_scatter_pa_compression_5(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScatterPaKvCacheLaunchData p;
    auto* dst=reinterpret_cast<uint64_t*>(&p);
    auto* src=reinterpret_cast<__gm__ uint64_t*>(tiling);
    for (int i=0;i<sizeof(p)/sizeof(uint64_t);++i) dst[i]=src[i];
if ASCEND_IS_AIV { if(p.indexBytes==4) RunCompression<uint16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunCompression<uint16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
}
