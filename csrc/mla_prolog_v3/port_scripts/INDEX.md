# port_scripts 索引

`csrc/mla_prolog_v3/port_scripts/` 里是这次移植的**全部改动脚本**（每个 `port_patchN.py`
都是一次可重放的小改动）以及两个构建辅助脚本。多数脚本已应用到当前代码；少数是**实测后
回滚的实验**，保留下来是为了让结论可复核（哪个方向试过、结果如何）。

| 范围 | 内容 | 当前状态 |
|---|---|---|
| `port_patch1..12` | 编译/接口适配：`Hcq=768` 放开、单入口 kernel + tilingKey 派发、GM→local tiling 拷贝、preamble 跳过 `ASCENDC_TPL`、dtype 名助手、平台信息、启动参数顺序、E8M0/FP8 dtype 映射 | **已应用** |
| `port_patch13..25` | 性能与正确性：tiling/workspace 缓存、无分配哈希 key、核数自适应、PA cache `stride0` 修复、`MAX_STEP`/`MM1_MIN_TILE` 旋钮、文档 | **已应用** |
| `port_patch26..27` | 在 DAV_3510 上放开 split-K/N 门禁 | **实验，已回滚**（设备报错 507015；且放开后 tiling 仍无候选）|
| `port_patch28..32` | roofline 记录、稳态 profile、mm1 最小瓦片默认 32、文档 | **已应用** |
| `port_patch33..36` | mm3 `baseK` 扫描（64/128） | **实测无变化，已回滚** |
| `port_patch37..38` | mm1/mm2 `baseK` 扫描（128/256/512） | **已回滚**（硬编码 256/512 直接触发 507015，证明 L1 顶格）|
| `port_patch39..55` | D=192 每步 128 token、`mla_prolog_v3_out`、PyBind、SGLang 兼容层（别名/路由/模式推断/ND→NZ/MXFP8 自动反量化）、文档 | **已应用** |
| `port_patch56..67` | SGLang 端到端：MLAPO 权重准备、scheme 状态对接、`forward` 改道尝试 | 大部分**已应用**；`forward` 强制改道**已回滚**（破坏 7/8 返回值解包契约）|
| `port_patch68..73` | 验收脚本、clean-build 记录、文档 | **已应用** |
| `port_patch74..84` | 主机侧 launch 描述缓存（含 tiling key 透出、thread-local 槽位、环形缓存） | **实测更慢（15.3 vs 11.8 µs），已回滚**；仅记录结论 |
| `port_patch85..91` | L1 容量证明、mm2 核数实验、文档一致性、从零复现记录 | 文档**已应用**；mm2 核数实验**已回滚**（<1%）|
| `port_gen_tiling_copy.py` | 生成 `mla_prolog_v3_tiling_copy.h`（逐字段 GM→local 拷贝） | 构建期工具 |
| `port_fix_merge_script.sh` | 让 CANN `merge_obj_text.sh` 幂等（增量构建必需） | 一次性前置 |

按主题回看结论，见同目录 `../README-port.md`（实现细节）与
`../REPORT-glm47-mla-prolog.md`（交付报告）。
