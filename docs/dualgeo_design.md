# DualGeo：面向空间推理的快慢双系统

> 研究设计 v0.2 · 2026-10-15
> 基座：GeoThinker 的 Qwen3-VL + 冻结 VGGT。
> 本仓库是 DualGeo 的唯一设计、实现与版本管理位置。

## 一句话

DualGeo 不再把 VGGT 视为每道题的固定前置步骤，而是建立两个职责明确的系统：**快系统**用模型内在的空间表征处理足够简单、可信的问题；**慢系统**只在预期收益覆盖成本时调用 VGGT，以多视角几何证据修正答案。

研究问题是：在同一骨干和预算下，能否让模型正确地区分“自己已能回答”和“值得请求外部几何”，从而得到优于常开/常关 VGGT 的准确率—成本折中？

## 为什么这样改

GeoThinker 的视觉 token 对 VGGT token 做 cross-attention，再以零初始化残差写回语言模型。这是获取外部多视角证据的有效方式，但当前路径会在融合之前完成 VGGT 编码；importance gate 也只是为全图 attention 的 key 加权，不能省去 VGGT 和全量 token 计算。

SpatioLM 与 3DThinker 提供了互补方向：前者说明可把深度、ray、camera 等空间先验蒸馏到可部署的 VLM 表征中；后者说明 VLM 可在训练期借助 VGGT teacher 学习几何想象。二者都不能替代测试时真实的多视角观测。因此它们最合适的定位是**快系统的能力来源**，而非另一条永远执行的外部几何路径。

DualGeo 首版不试图同时解决局部裁剪、目标跟踪、时空 tube、多次恢复与复杂 RL。那些是慢系统被证明有价值之后的效率优化，不能掩盖核心因果问题。

## 系统定义

```text
输入图像/视频 I 与问题 q
          │
          ▼
快系统：Qwen 视觉 token + 轻量内在空间 adapter
          │
          ├── fast state：空间 predicate、答案风险 u_fast、问题表征
          └── gate（只读取调用 VGGT 前已有的信息）
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       SELF                SEEK
   直接生成答案        冻结 VGGT（固定帧预算、全图）
                            │
                            ▼
              GeoThinker cross-attention 残差融合
                            │
                            ▼
                         慢系统答案
```

### 快系统（Fast / SELF）

快系统只使用 Qwen 已有的视觉/文本特征和小型可训练 adapter，不实例化、不加载、更不运行 VGGT。它输出：

- 用于回答的内在空间表征；
- 可选的相对深度、方位、运动等 predicate 辅助预测；
- 经独立验证集校准的答案错误风险 `u_fast`；
- 给 gate 使用的问题—视觉摘要。

训练期可用 DA3 的 depth/ray/camera target，以及冻结 VGGT 的压缩关系 target 做蒸馏；部署 checkpoint 不包含这些 teacher。首版的重点不是声称快系统恢复了完整 3D，而是验证它能否可靠处理不需要外部多视角证据的题。

### 慢系统（Slow / SEEK）

慢系统复用 GeoThinker 的冻结 VGGT、二维网格对齐、geometry merger 和零初始化的 cross-attention 残差。VGGT 只接收 gate 选中的样本；首版固定使用预处理时的同一组全图帧（例如 4 帧），不引入 ROI crop 或可变帧数，确保慢系统收益和成本可比较。

慢系统不是总比快系统正确：低纹理、强反光、镜头切换或帧重叠不足时，外部几何可不可靠。因此训练与评估都须保留“调用无益或有害”的样本，融合残差从零开始学习，而非无条件覆盖基座表示。

### Gate：按净收益选择慢系统

gate 在看不到 VGGT 输出时，对两个动作估计条件风险和成本。其部署规则为：

$$
\operatorname{SEEK}(x)=\mathbb{1}\left[
\widehat{R}_{\mathrm{fast}}(x)-\widehat{R}_{\mathrm{slow}}(x)
> \lambda\,\widehat{C}_{\mathrm{VGGT}}(x)+\delta
\right].
$$

其中 `x=(I,q)`；`R_fast` 与 `R_slow` 是答案损失或错误概率，`C_VGGT` 是目标硬件实测的延迟/显存成本，`δ` 是抑制误调用的安全余量。`R_slow` 不是在部署期先跑 VGGT 才得到，而是由离线反事实路径学习：对同一训练样本缓存 SELF 和 SEEK 的结果，监督 gate 预测慢系统相对快系统的净收益。

更直接地，对动作 `a∈{SELF, SEEK}` 定义训练标签：

$$
y_a(x)=-\ell_a(x)-\lambda C_a(x),
\qquad
\text{gate 学习排序 } y_{\mathrm{SEEK}} \text{ 与 } y_{\mathrm{SELF}}.
$$

只有 `SEEK` 样本才可进入 VGGT provider。混合 batch 必须按动作分桶；不得为了张量对齐而暗中为 `SELF` 样本执行 VGGT。

## 首版范围与非目标

| 首版必须完成 | 明确延后 |
|---|---|
| 二元 `SELF / SEEK` gate | `LOCAL / GLOBAL` 多档动作 |
| 固定帧数的全图 VGGT | ROI、crop、tube、目标跟踪 |
| 一个内在空间 adapter 与校准风险 | 完整 SpatioLM 3R block 的复刻 |
| GeoThinker 已有 cross-attention 慢融合 | 动态注入层、复杂 source attention |
| 反事实价值标签与真实成本测量 | RL、多轮检索/恢复 |

这不是放弃局部几何。若首版显示 `SEEK` 确实能在相同成本下提高表现，第二阶段才测试“先选帧/区域，再调用 VGGT”。届时必须以同一题的全图 VGGT 为上界，验证 crop 是否因内参变化、上下文丢失或 correspondence 破坏而降低几何质量。

## 训练与实验

1. **基线与成本表**：固定 Qwen3-VL、VGGT、帧数和分辨率，测量 `SELF`、always-SEEK 的准确率、p50/p95 延迟、峰值显存；缓存二者的逐样本损失与结果。
2. **训练快系统**：冻结主干，训练内在 adapter/predicate head；在未参与训练的验证集做 temperature scaling 或 isotonic calibration，报告 ECE、Brier 与 risk–coverage。
3. **训练慢融合**：冻结 VGGT，训练 GeoThinker merger/cross-attention；确认 always-SEEK 在需要多视角的分桶上有独立收益，且坏几何不会稳定伤害基座。
4. **训练 gate**：以缓存的反事实净价值训练二元分类/排序头；它的输入不得包含 VGGT feature、pose、慢系统 loss 或任何调用后的质量指标。
5. **端到端评估**：在固定预算下比较 learned gate、always-SELF、always-SEEK、oracle gate 与简单 entropy heuristic。

成功不以平均准确率单独判定。至少应同时证明：快系统本身有校准后的有效覆盖；learned gate 的 utility 高于常开和常关；`SELF` 路径的 profiler 证实未加载/运行 VGGT。

## 最小消融

- Base Qwen3-VL（无空间 adapter、无 VGGT）
- Fast only（内在空间 adapter、无 VGGT）
- Always SEEK（GeoThinker 慢系统）
- Fast + Always SEEK（验证快系统与慢融合能否共存）
- Fast + learned gate
- Fast + oracle gate（可达到的上界）

按单图/视频及 relative depth、方位、接触/拓扑、运动、cross-view 等类型分桶。报告 accuracy、NLL/Brier/ECE、seek precision/recall、oracle regret、VGGT 调用率、p50/p95 延迟、显存与每正确题成本。

## 开发入口

实现步骤、接口约定、GeoThinker 的具体改动位置和 smoke test 见 [开发蓝图](dualgeo_architecture.md)；当前实现状态、版本记录和接力规则见 [开发日志](dualgeo_development.md)。
