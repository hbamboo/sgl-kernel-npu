# MLA Prolog V3 decode：576维直写与小batch优化验证

576维Query直写、SGLang FlashMLA前消除Query concat及小batch的1:1混合核优化已实现。精度验收通过；性能有效覆盖16/20个组合。direct-cache batch7/8/9/10尚未完成独占性能计时，不能将本记录当作20/20验收通过。剩余补测已按用户要求暂停，等待后续明确命令。

## 实现

原生代码提交 `0d19ea0d6cfa39afe72446f494cdf86d3a1dd174`，576维直写前序提交86b4069。`packed_query=True`分配一个[T,20,576] BF16 tensor，q_nope和q_rope分别为其512/64维视图。Host传递物理head stride，Qn Fixpipe与RoPE最终搬出直接写入对应位置。SGLang的`combine_query`通过共享存储、偏移及stride检查恢复完整连续视图，保留五输出接口及旧独立tensor的兼容路径。

精确GLM mode0、norm输出、规则576步长、二维T1–10且物理核数足够时，选择独立的20核AIC:AIV=1:1入口。逐T固定几何，T1–3关闭head-local，T4–10启用完整head同核生产消费。其它输入保持原入口。两入口在独立AscendC target中编译，ELF元数据分别为1:2和1:1，避免同编译单元覆盖物理比例。未扩展GM tiling结构ABI。

`MLA_PROLOG_V3_DECODE_MIX11=0`关闭新1:1入口，`MLA_PROLOG_V3_DECODE_STATIC=0`关闭小batch静态入口，`MLA_PROLOG_V3_DECODE_TUNE=0`关闭自动调优。显式调试覆盖须清除后再复现默认策略。SGLang的`SGLANG_NPU_MLAPO_V3_PACKED_QUERY=0`可切回分立输出。此次只优化decode；prefill不使用该算子。

## 验证与性能口径

- 480个真实checkpoint组合与原路径逐位一致：层0/2/23/46、3个seed、T1–10、两种cache布局、两种epsilon及不同输入/权重尺度。候选前恢复初始cache，核对完整cache和canary。
- 28项一键验收通过，含CPU golden；95个形状（T1–92、127/128/129）及5个norm关闭边界通过。
- 20种T/布局的捕获图各更新输入与slot并回放3次，回放前恢复cache；混合T1/T9多流和608步长padding canary回退通过。
- 当前SGLang `opt/p_kv_shared_tune` 56项接入测试通过，含分配式wrapper、图捕获、禁止Query concat、真实FlashMLA以及T9/10/11边界。功能提交bb3683a，边界测试提交7d9cdf6a32。

物理NPU7（Ascend950DT），CANN9.1.0，编译目标Ascend950PR_9599。checkpoint为`/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`。性能使用第2层BF16准备权重、seed42、两个独立常驻进程中的原始二进制与候选，31组交替A/B/A、B/A/B，每图47次调用，每路取NPU event中位数。计时前输出、Qnorm与完整KV/KR cache对原始库逐位核对。有效组合在开始及组内10/20/30轮采样时只出现两个测试PID；发现其它进程则整组作废。进程抽样不能证明抽样点间绝无短任务，原始样本及审计均保留。

这是prolog+Query组装的图回放微基准，不包含FlashMLA计算、完整模型TPOT或QPS。加速比为原耗时/新耗时；性能提升30%表示比值≥1.30，不能与时延下降30%混用。业务日志108条周期采样主要为batch1–10，因此完整验收仍需补齐direct-cache的7–10，不能用大batch替代。周期采样不是每步的完整频率分布。

| Cache布局 | Batch | 原始+concat µs | 新packed µs | 加速比 | 时延下降 |
|:---|---:|---:|---:|---:|---:|
| scratch | 1 | 20.156 | 14.342 | 1.405x | 28.84% |
| scratch | 2 | 20.293 | 14.275 | 1.422x | 29.65% |
| scratch | 3 | 20.498 | 14.650 | 1.399x | 28.53% |
| scratch | 4 | 20.658 | 14.734 | 1.402x | 28.68% |
| scratch | 5 | 20.732 | 14.987 | 1.383x | 27.71% |
| scratch | 6 | 20.784 | 15.048 | 1.381x | 27.60% |
| scratch | 7 | 20.861 | 15.224 | 1.370x | 27.02% |
| scratch | 8 | 20.925 | 15.232 | 1.374x | 27.21% |
| scratch | 9 | 20.940 | 15.087 | 1.388x | 27.95% |
| scratch | 10 | 20.914 | 15.040 | 1.391x | 28.09% |
| direct | 1 | 20.226 | 14.288 | 1.416x | 29.36% |
| direct | 2 | 20.243 | 14.247 | 1.421x | 29.62% |
| direct | 3 | 20.317 | 14.410 | 1.410x | 29.07% |
| direct | 4 | 20.574 | 14.844 | 1.386x | 27.85% |
| direct | 5 | 20.737 | 15.046 | 1.378x | 27.44% |
| direct | 6 | 20.753 | 15.080 | 1.376x | 27.34% |

有效16组合加速37.0%–42.2%，时延下降27.0%–29.7%。95形状扫描等套件中的辅助计时未作性能验收依据。补测曾遇其它进程干扰，另有两个工作进程等待时收到来源不明SIGKILL；已丢弃受干扰组，保存此前有效行。后续等待卡空闲满1800秒后退出，rc=1是资源等待超时，未发生数值校验失败。

## 量化与探索

checkpoint的q_a_proj、q_b_proj、kv_a_proj_with_mqa三份projection为F8_E4M3加每32元素U8 scale，kv_b_proj为BF16。框架融合q_a与kv_a，因而表现为两组量化线性层。现有V3准备阶段反量化为BF16并运行mode0，本次保持该策略。

保留576直写、逐T静态几何、独立1:1物理入口和20核完整head切分。Qn共享L1-A配合wide-N、完整K192/宽L0B、固定buffer替代TPipe、其它核数组合等原型未稳定胜出，已从正式代码移除。设备objdump元数据可用，但本环境指令反汇编为`<not available>`，未据此声称定位唯一指令瓶颈。后续可进一步验证小矩阵初始化、同步及重复加载成本；native MXFP8小batch需要独立评估精度和准备开销，不计入当前收益。

## 可追溯记录

本目录的`validation/compiled_mix11_20261031`保存有效原始样本、PID审计、数值回归、验收日志、元数据、源文件哈希和暂停状态；`summary.json`明确标记完整性能验收未完成。未提交动态库、模型权重或大型profiling文件。原始脚本和完整历史日志保留在`/tmp/mla-output576-20261030`。

已测动态库SHA256：`9fc6a24c417ae500b628ac351b12cd22c5c2e1f14f82ffc68d73e19c924d59b9`。源码包内当前库与该哈希一致，阶段2备份为`/tmp/mla-output576-20261030/phase2-packed-query.so`。pybind能力为packed_query_version=1、small_decode_version=2、small_decode_cv_ratio=1；ratio表示专用入口可用性，调用是否命中取决于形状和开关。全局安装包未替换。

归档的文本日志仅规范化行尾空白；原始文本保留在上述临时目录，JSON原始样本未作此处理。
