# ScatterPaKvCache 迁移、融合及验收

已交付到指定目录，并在物理 7 号 Ascend950DT 上完成正常包导入测试和指定权重的 SGLang 验证。

## 交付位置

- Kernel：/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/scatter_pa_kv_cache
- Python：/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/scatter_pa_kv_cache.py
- SGLang：/sgl-workspace/sglang-p-kv-opt，基于 opt/p_kv_shared_tune。
- 权重：/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8。
- 原始日志、JUnit、profile、交付哈希与验收审计：/mnt/paas/hxdnpu/sgl-kernel-npu-scatter/scatter-results。

## 实现

完整保留源算子文件，原始 A5 kernel 类不改算法，40 个 tiling 字段的类型和顺序一致，tiling 计算公式一致；调整注册上下文、平台查询及当前流启动胶水。正常 CMake 构建生成伴随库，由包初始化加载，调用方式为 Python wrapper / torch.ops.npu。支持 mutation alias、Meta、非默认 stream 与 graph replay。

新融合入口直接从 [T,20,448] 投影及已旋转的 [T,1,64] K-RoPE 写最终 paged K/V，避免 V 连续化和完整 K 拼接。Prefix 批次每层写一次；无 prefix 保留连续 FIA 输入，使用联合 K/V scatter。latent sink 与 PD buffer 布局保持原样。SGLANG_NPU_PREFILL_FUSED_KV_SCATTER=1 控制接入，默认关闭便于 A/B。

修复了移植时需要保留的真实数值类型压缩、混合 K/V PA_NZ、FP4 x2 逻辑维度。另修复闲置 host 字段未初始化导致序列化 graph 缓存键变化的问题；只初始化适配器成员，不改源公式。CANN FP8 vector Cast 在 Cube dummy 编译上触发编译器错误，入口使用 SDK AIV 编译条件避免无关实例化。

## 验证与性能

精确目标目录通过正常包导入的 **283 项测试全部通过**，覆盖原路径、压缩、混合类型、FP4 位拷贝、尾块、非连续布局、特殊 stride、stream、Meta 和 graph replay。模型预热后交替运行 10 组 A/B，所有 logits 逐位一致。

| 口径 | 原路径 | 融合路径 | 改进 |
|---|---:|---:|---:|
| BS12，33305 新 token，prefix128/请求，完整 prefill 前向中位 wall time | 788.441 ms | 673.555 ms | 耗时下降 14.57% |
| 单层布局整理+scatter，10 组设备 kernel 序列中位数 | 2847.153 us | 418.954 us | 6.80× |
| 同模型 profiler 的 cache 写入 kernel 总时长 | 102.795 ms /94次 | 20.376 ms /47次 | 每层一次融合写入 |

Standalone 最低有效读写带宽 **3.063 TB/s**，同卡大块连续 copy **3.302 TB/s**，达到该实测参照的 **92.7%**。扫描了 72 个 tile/core/coalescing 配置，覆盖顺序 slot、打散 page 和随机 slot，默认 tile4、64 AIV 核处在带宽平台附近。模型计时与 profiler 采样分开，未丢弃样本或择优。

这不是 HTTP TTFT 或跨卡 PD 服务吞吐结果；带宽按逻辑最低读写字节计算，不是硬件 HBM counter，也不宣称已证明绝对全局最优。当前目标平台为 Ascend950；源文件保留其他架构实现。PyTorch 包暴露当前 PyTorch 可表达的 dtype，CANN-only HiFloat8/FP4 E1M2 原始声明仍保留，未伪装成其他数值类型。

## 使用

在原有 expanded prefill 配置上设置 SGLANG_NPU_PREFILL_FUSED_KV_SCATTER=1。标准构建、API 与测试命令见 README-port.md。未全局安装包、未修改原始 ops-transformer 源码；已有 MLA/FlashMLA 改动被保留。代码为工作区改动，未代为提交或推送。
