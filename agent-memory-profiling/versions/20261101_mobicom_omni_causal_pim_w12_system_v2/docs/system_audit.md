# HyperPIM MobileMem-Omni 系统审计（2026-10-26）

## 数据与协议

- 原始数据为 `MobileMem-main/data/omni/data.jsonl`、`questions.jsonl` 和 `image.zip`。
- 当前发布包含 20 个用户、9,308 个问题；官方七类任务分布在问题文件中。
- Omni 没有每题 release timestamp，因此主实验固定为 `history_end`：同一用户完整可见历史，不宣称在线时间回放。
- `scripts/omni_prepare.py` 将 serving manifest 与 evaluation-only labels 分离；serving 入口只读取 `omni_graph.json`、`manifests.jsonl`、split/cohort 元数据。

## 可复用实现

- `omni_prepare.py`：会话、turn、图片引用和来源身份校验。
- `omni_proof_sources.py`：原始文本对象、原始图片对象、来源容器和唯一物化对象。
- `extract_omni_visual_locators.py`：无 QA 输入的 Qwen3.5-VL-4B 离线视觉 locator；locator 只用于定位，Reader 仍读取原图。
- `build_omni_proof_index.py`：统一文本嵌入空间中的 atom/region 索引。
- `omni_proof_join.py`、`omni_proof_grounding.py`：需求绑定、来源见证和共享对象去重。
- `omni_proof_lifecycle.py`：按唯一原始对象记录 Reader prefill 的生存期与物化前沿。
- `pim_backend/lpddr_pim.py`：共享 LPDDR 时序与候选 PIM 扫描模型，明确标记为架构估算。

## 关键缺口与修正

1. 旧主线只授权 development 用户；现已增加显式 split 授权，默认调用仍只允许 development，主实验配置明确列出三个 split。
2. 旧索引构建硬编码 `evidence_compiler_index_dev3_20261024_v3` 的图片成本；现改为从同一冻结 locator 记录的 processor token 成本读取，避免跨版本成本污染。
3. 旧 v1/v2 将 facility-location 代理和跨模态父边排序当作主假设；新主线只把 embedding 当候选生成，把来源绑定、AND/OR 见证和唯一对象物化作为实际决策接口。
4. 主实验 Judge 固定为用户指定且本地可核验的 `Qwen3-VL-8B-Instruct`；它作为本实验统一 Judge 使用，不冒充官方 Qwen3-14B 分数。主实验仍需在全量 Reader 结果完成后实际运行 Judge 才能闭环。
5. AttAcc checkout 实际可核验的是 LPDDR5 DRAM 读写和 host trace；当前 LPDDR-PIM 参数仍为未器件校准的扩展模型，不将其写成手机实测或端到端加速。

## 资源与入口

- Reader：本地 Qwen3.5-VL-4B，7 张空闲 RTX 3090 可用于用户级并行。
- Judge：配置锁定为本地 `Qwen3-VL-8B-Instruct`，缺失时显式失败/挂起，不切换模型。
- 主入口：`scripts/run_main_experiment.py`；评测入口：`scripts/evaluate_main_experiment.py`；PIM 账本：`scripts/profile_lpddr_pim.py`。

## 主线边界

不启动外部 baseline、消融矩阵或参数搜索。开发队列历史结果只作失败归因；本版本一次性冻结数据、模型、提示词和预算，随后在同一实现上运行完整用户集合。
