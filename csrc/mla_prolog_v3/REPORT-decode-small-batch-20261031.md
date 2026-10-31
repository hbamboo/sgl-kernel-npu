# MLA V3 decode：业务小 batch 特化验证

576维直写和FlashMLA前去concat已完成。继续针对现有业务日志主要出现的batch1–8优化：**当前尚未达到30%目标，目标保持进行中。** 此报告只讨论decode。

## 实现

在已验证的阶段1提交86b4069上，增加受严格形状约束的BF16小batch路径：He2048、Hcq768、Hckv512、20heads、D192、Dr64、norm输出、packed Query、16核及原有MM切分全部匹配才选择。T1..8分别实例化，固定几何交给编译器；epsilon、缩放、cache页大小/物理步长、索引和输入值仍是运行时参数。T选择编码在现有tilingKey高32位，避免额外读取选择字段；没有新增tiling结构ABI。

未匹配的输入沿用原路径。`MLA_PROLOG_V3_DECODE_STATIC=0`可单独关闭该路径；`MLA_PROLOG_V3_DECODE_TUNE=0`关闭所有自动decode调优。显式STATIC=1可用于消融。Pybind提供`mla_prolog_v3_small_decode_version=1`。

## 性能口径与实测

物理7号卡，CANN9.1.0/Ascend950PR_9599，真实checkpoint第2层BF16准备权重，同种子输入。原始库为保存的cache-only baseline-cache.so；新库3ccd5fa76b896c875b142ed466452f88107669aa7e69b6272a4cef37eda941a3256：`3ccd5fa76b896c875b142ed466452f88107669aa7e69b6272a4cef37eda941a3`。两库分属两个常驻进程，输出与完整cache先逐位核对，然后紧密交替A/B/A与B/A/B，31组，每图47次调用，取每路中位数。每个batch开始及组内抽样核对7号卡进程列表，采样只看到了两个测试进程。

| Batch | 原始库+concat µs | 新库packed µs | 加速比 | 时延下降 |
|---:|---:|---:|---:|---:|
| 1 | 20.297 | 16.943 | 1.198x | 16.53% |
| 2 | 20.232 | 16.848 | 1.201x | 16.73% |
| 3 | 20.551 | 17.342 | 1.185x | 15.61% |
| 4 | 20.632 | 17.247 | 1.196x | 16.41% |
| 5 | 20.561 | 17.361 | 1.184x | 15.56% |
| 6 | 20.758 | 17.400 | 1.193x | 16.18% |
| 7 | 20.861 | 17.604 | 1.185x | 15.62% |
| 8 | 20.871 | 17.447 | 1.196x | 16.41% |

这是算子与Query组装的微基准，不是整网TPOT/QPS。两进程分别持有权重，缓存行为不能等同于单进程整网。原有业务日志108条周期采样中100条为batch1–8，其余为9/10；周期日志不能当作每步的完整频率分布。

先做的独立进程A/B/A，A2在T1..4出现严重耗时尖峰，**该轮不用于性能验收，也不将A2和A平均来计算加速**。保留其原始JSON用于数值检查及问题追溯。没有用受干扰的慢基线宣称30%达标。

## 验证

* 原有及阶段1新增一键验收28项通过，覆盖CPU golden、真实权重、多几何、边界、out缓冲、多流及图捕获。
* 新增384组合：真实层0/2/23/46，seed0/42/2026，T1..8，两种cache布局，两种epsilon及相应scale。候选前恢复初始cache，五路非空输出、完整cache和canary逐位一致，避免漏写被已有参考值掩盖。
* T1..92加127/128/129共95形状通过；query_norm_flag=False的5边界通过。
* 在SGLang提交bb3683a的隔离worktree上47项接入测试通过，覆盖实际wrapper分配/图捕获、scratch与direct-cache、禁止Query concat和真实FlashMLA结果一致性。

## 本轮其他尝试

CQ/KV共享完整L1输入只带来小幅收益；down-K256与QC-N128/K128分块、空闲向量核快速路径、8/10/12/14/15核及单一动态T几何特化均未稳定优于保留的逐T特化，已移出候选。历史patch与原始数据保留用于复查，不能把其受干扰行当作有效加速结论。

npu-objdump元数据和AIC/AIV符号/哈希已保存，但本机llvm-objdump输出全部为`<not available>`，没有有效设备指令反汇编；未据此宣称定位到唯一瓶颈。没有修改SDK或全局安装包。

## 复现与下一步

使用匹配的源码包和编译库，SGLang保持packed Query与DECODE_TUNE启用。可运行`tests/test_small_static_regression.py`；`tests/final_compare.py`设置`SIZES=1,2,3,4,5,6,7,8`可做同库消融。同库消融不替代保存原始二进制的交叉对照。

原始日志在`/tmp/mla-output576-20261030`，主要结果随本提交保存在`validation/compiled_smallbatch_20261031`。后续优先检查Qn在多个N块间重复加载相同A，以及L1/小矩阵初始化开销；这些只是待验证方向，不承诺收益。达到业务常见batch口径30%前不宣布目标完成。
