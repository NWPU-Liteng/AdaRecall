# HyperPIM 主系统设计

## 核心对象

系统维护两种互补表示：

- **来源证明超图**：需求节点、原始 turn、原始图像、locator 记录和 `AND/OR/关系` 边。locator 是可检索描述，不是答案事实；所有最终证据必须回到原始 text/image 对象。
- **物理检索索引**：每个原始对象只保存一份文本向量或图片对象身份，按 atom/region 形成连续向量块。多个证明共享对象时只收费一次物化成本。

## 查询路径

1. Qwen3.5-VL-4B 只依据问题生成结构化 source-fact/operation contract；解析失败退回完整原问题软检索。
2. 文本 turn 与视觉 locator 在同一冻结文本嵌入空间检索，避免 text-image cosine 直接竞争。
3. 候选通过来源见证校验：变量绑定、同/异实体关系、原始引用和时间范围必须一致；不能证明的开放问题保留为 contextual package，不伪造 AND。
4. 见证搜索以最弱需求支持优先、唯一物理对象成本次优先；最终用 tokenizer/processor 重新物化并检查真实输入预算。
5. Reader 接收原始对话和必要原图。locator 明确标记为定位提示，不进入事实替代链。
6. 生命周期账本记录对象 materialize、Reader prefill、decode、release；同一执行轨迹生成 PIM profile 输入。

## PIM 路径

主系统的 Reader 和线上检索仍走 Host。`profile_main_pim.py` 从已完成请求中重放原始 retrieval views，并对每个 query vector 比较 Host FP32 Top-48 与按全局对称尺度量化的 LPDDR-PIM INT8 Top-48；它记录候选重合、服务端 CPU 扫描时间、索引量化/常驻容量，以及同一扫描形状的 Host/PIM 周期模型。PIM 的数值路径是 INT8 软件功能模拟，周期路径是未器件校准的 LPDDR5-6400 bank 模型；二者都不是 AttAcc 上已经验证的 LPDDR-PIM 指令执行。模型结果只覆盖向量点积与分银行 Top-K，不包含 parser、reranker、超图见证构建、Reader prefill/decode，也不构成端到端 speedup。

PIM 候选需满足 PIM 向量 Top-48 与 FP32 Host 排名质量检查，并单独报告一次性 INT8 索引装载成本及 warm-resident 扫描成本。容量 admission 仅针对向量索引，不代表 Qwen Reader 权重可驻留。LPDDR 时序、bank 内部 MAC、命令/fence、host 服务率和刷新均为固定架构假设；未做设备校准、共享总线并发或手机实测时，只能称为模拟估算。

## 可验证主张

- 证明超图实际改变证据恢复与原始对象物化，而不是只保存 session 字段。
- 视觉 locator 的命中可以沿 provenance 边回到原图，且图片成本由实际 processor token 计费。
- locator 输出允许 EOS 缺失和像素框格式；构建阶段逐个恢复完整 region，并按原图尺寸归一化，不因截断 JSON 直接丢弃视觉证据。
- Reader 输入、规划时间、生命周期前沿和 PIM trace 来自同一请求记录；向量扫描收益与在线端到端时延分开报告。
- 质量结果只有在 evaluation-only 标签、官方 Judge prompt 和固定本地 `Qwen3-VL-8B-Instruct` 实际运行后才闭环；该结果不表述为官方 Qwen3-14B 分数。
