# DualGeo 开发文档：在 GeoThinker 上实现快慢双系统

> 目标分支建议：从 M0 基线提交新建 `dualgeo-fast-slow`。
> 首个目标：实现可测量、可关闭的二元 `SELF / SEEK` 路径；不要先实现 ROI 或视频 tube。

## 1. 改造原则

1. **快路径是实路径。**`SELF` 时进程不得构造 VGGT 模型、加载 VGGT 权重或调用 `encode`。仅仅让残差为零不算跳过。
2. **慢路径复用而不重写 GeoThinker。**首版保持冻结 VGGT、`GeometryFeatureMerger`、二维对齐和 `QwenVGGTInteractionv2Flash` 的语义；只把它们放到明确的 `SEEK` 分支。
3. **gate 没有未来信息。**gate 的输入仅限 Qwen 的视觉 token、问题 token、视频元数据和成本表。VGGT feature、camera token、慢路径 loss、外部质量分数只能在 `SEEK` 后使用。
4. **先二元，后稀疏。**第一版固定全图和 `k=4` 帧。帧选择、ROI、crop 和恢复是后续实验，不能混入 MVP。
5. **保持 Qwen 主干最小侵入。**控制器、路由与训练损失放在 `qwen_vl/dualgeo/`；`modeling_qwen3_vl.py` 只增加显式的 hook/输出，不散布策略逻辑。

## 2. 已核对的 GeoThinker 基线

本计划以 Qwen3-VL 路线为准，主要入口如下。

| 现有位置 | 已有能力 | DualGeo 的处理 |
|---|---|---|
| `src/qwen_vl/model/qwenvl3/modeling_qwen3_vl.py` | 创建 `geometry_encoder`，调用 `_process_geometry_features`，把 `geo_embeds` 送入 decoder。 | 重构为显式可选的 slow provider；快路径必须可传 `None`。 |
| 同文件的 `Qwen3VLTextDecoderLayer` | 在配置指定的层调用 `cross_attn`，并把残差写回 image token。 | 首版保留注入位置与零初始化行为；不做 layer gate。 |
| `src/qwen_vl/model/qwen_interaction.py` | `QwenVGGTInteractionv2(Flash)` 实现 image query → VGGT key/value；importance gate 仍对全 token attention 加 bias。 | 作为 slow-fusion 基线；不把 importance gate 当作按需调用机制。 |
| `src/qwen_vl/model/geometry_encoders/vggt_encoder.py` | 冻结 VGGT wrapper。 | 仅在 `SEEK` worker/model 实例中初始化。 |
| M0 的 `src/qwen_vl/dualgeo/` | 严格反事实缓存、路由类型、计时与独立加载器。 | 作为快慢路径分离与后续 gate 训练的唯一公共接口；不再依赖 3DThinker 实现。 |
| `src/qwen_vl/train/argument.py`、`train/train_qwen.py` | GeoThinker/Qwen3-VL 的参数解析、模型装配、冻结策略。 | 增加 DualGeo 参数、trainer 和 checkpoint manifest。 |

当前 `modeling_qwen3_vl.py` 在处理 image placeholder 时会访问 `geometry_encoder_inputs[0]`；因此不能只把 `use_geometry_encoder=False` 后继续使用同一个 GeoThinker model 类。M0 必须先把 placeholder mask 与 geometry 输入解耦，并为 `geometry_encoder_inputs=None` 添加单测。

## 3. MVP 架构与数据契约

### 3.1 两个可部署 checkpoint

为确保 SELF 真正不占用 VGGT 显存，首版导出两个 checkpoint/服务实例：

```text
dualgeo-fast
  Qwen3-VL + FastSpatialAdapter + DualGeoGate
  无 geometry_encoder state_dict

dualgeo-slow
  Qwen3-VL + 同一 FastSpatialAdapter + GeoThinker merger/cross-attn + frozen VGGT
```

两者共享 Qwen 与 fast-adapter 权重。部署控制器先向 fast 实例请求 `RouteState`；只有动作是 `SEEK` 才把原始已选帧送到 slow 实例。这样比在同一模型中惰性 if/else 更容易用显存与调用日志验证约束。训练期可使用一个包含 VGGT teacher 的进程或离线 cache，但不能将其作为 SELF 导出物。

### 3.2 核心结构

```python
@dataclass
class RouteState:
    question_state: torch.Tensor       # [B, D]
    spatial_state: torch.Tensor        # [B, D]
    answer_risk: torch.Tensor          # [B], calibrated after training
    video_meta: torch.Tensor           # [B, M]

@dataclass
class RouteDecision:
    seek: torch.BoolTensor             # [B]
    predicted_gain: torch.Tensor       # [B]
    predicted_cost_ms: torch.Tensor    # [B]

class FastSpatialAdapter(nn.Module):
    def forward(self, visual_tokens, question_tokens, video_meta) -> RouteState: ...

class BinaryEvidenceGate(nn.Module):
    def forward(self, state: RouteState, cost_table) -> RouteDecision: ...

class DualGeoController:
    def generate(self, batch, budget) -> GeneratedAnswer: ...
```

`question_state` 可以先由非 padding 的问题 input embedding mean-pool 得到；`spatial_state` 由 Qwen visual token 的轻量 adapter/attention pool 得到。首版无需为获得 gate 再额外运行完整自回归答案。`answer_risk` 可由 fast answer head 的 temperature-scaled probability 或单独 risk head 给出，但只在验证集拟合温度。

### 3.3 训练样本与反事实缓存

每条样本固定 `sample_id`、问题、相同的预处理帧及其原图坐标/尺寸。训练 gate 需要以下记录：

```json
{
  "sample_id": "...",
  "frame_ids": [0, 8, 16, 24],
  "fast_loss": 0.0,
  "slow_loss": 0.0,
  "slow_latency_ms": 0.0,
  "slow_peak_mem_mb": 0.0,
  "fast_correct": true,
  "slow_correct": true
}
```

净收益标签采用：

$$
g(x)=\ell_{\mathrm{fast}}(x)-\ell_{\mathrm{slow}}(x)
-\lambda_t T_{\mathrm{slow}}(x)-\lambda_m M_{\mathrm{slow}}(x).
$$

训练时可只缓存慢路径输出/损失，避免每个 step 跑 VGGT。验证和测试按原始题目切分，不能让同一视频的相邻片段同时落入训练与测试。

## 4. 建议的文件与职责

在 `src/qwen_vl/` 新增：

```text
dualgeo/
  __init__.py
  config.py          # DualGeoConfig，动作、成本、checkpoint 版本
  fast.py            # FastSpatialAdapter、风险/predicate head
  gate.py            # BinaryEvidenceGate 与阈值决策
  controller.py      # fast → 分桶 → slow → 结果回填
  losses.py          # fast 蒸馏、风险、gate utility loss
  cache.py           # 反事实 JSONL/Parquet 读写与 schema 校验
  metrics.py         # ECE、risk-coverage、utility、regret、调用率
  types.py           # RouteState、RouteDecision、诊断结构
```

改动现有文件应限定为：

| 文件 | 最小改动 |
|---|---|
| `model/qwenvl3/modeling_qwen3_vl.py` | 解耦 image/video placeholder 与 `geometry_encoder_inputs`；提供不含几何的视觉 token hook；slow forward 明确要求 geometry input。 |
| `model/qwen_interaction.py` | 不修改 attention 数学；补充接受/检查 `geo_embed_mask` 的 shape 与零 token 行为。 |
| `train/argument.py` | 增加 `dualgeo_enabled`、`dualgeo_mode={fast,slow,gate}`、成本权重、gate margin、cache 路径等参数。 |
| `train/train_qwen.py` | 按 mode 装配 fast 或 slow 模型并设置冻结策略；fast checkpoint 不创建 geometry encoder。 |
| `train/dualgeo_trainer.py`（新增） | 组合语言、蒸馏、风险及 gate loss；读取离线反事实 cache。 |

不要将 `controller.py` 的路由决策写进 `Qwen3VLTextDecoderLayer.forward`；该层只负责已给定的几何残差注入。

## 5. 分阶段实现清单

### M0：拆开并测通原始路径

- 固定一个 Qwen3-VL、VGGT、处理器、4 帧规则和开发集。
- 跑通 GeoThinker always-SEEK 与纯 Qwen always-SELF；保存逐样本 answer/loss、`torch.cuda.max_memory_allocated()`、端到端延迟。
- 修复 `geometry_encoder_inputs=None` 时的 forward；新增 `test_no_geometry_forward`。
- 验收：SELF 输出与原生 Qwen 在相同权重和输入下数值一致（允许 bf16 容差）；SELF profile 无 VGGT module/call。

### M1：快系统能力与校准

- 新建 `FastSpatialAdapter`，初版采用视觉 token pool + 两层 MLP；可附带 relative-depth/关系 head。
- 训练 adapter 与 head，冻结 Qwen；DA3/VGGT 只作为训练 teacher 或 cache。
- 在独立 calibration split 拟合温度；保存温度与数据版本到 manifest。
- 验收：fast-only 不低于 Base 的目标 probe，且在 `risk <= τ` 的 coverage 内错误率和预测风险一致；报告 ECE/Brier。

### M2：慢系统兼容性

- 保持冻结 VGGT，复现 always-SEEK GeoThinker 基线；逐样本记录 slow loss 和真实成本。
- 将 fast adapter 加到 slow checkpoint，但不改变原 GeoThinker fusion，先确认不会破坏 always-SEEK。
- 验收：slow 与基线的输出/指标可复现；全图网格对齐无一维 token 插值。

### M3：二元 gate 与控制器

- 从 M0/M2 cache 生成 `g(x)`；训练 `BinaryEvidenceGate` 做排序或 margin classification。
- `DualGeoController` 先跑 fast probe，按 `seek` 分桶，再只将 seek 子 batch 转交 slow 实例，最后以原顺序合并答案。
- 设置硬预算：每题最多一次 slow 调用，超过总延迟/显存预算强制 SELF。
- 验收：日志中 SELF 的 `vggt_calls=0`；learned gate 在同一成本下胜过 always-SELF/entropy heuristic，或在相近准确率下降低调用率。

### M4：仅在 M3 成功后

测试多档帧数，再测试 frame selection；最后才测 ROI+margin+global anchor。任何 crop 路径都须记录可逆坐标变换，并与相同帧的 full-frame VGGT 对照几何误差、答案与真实成本。

## 6. 训练与推理伪代码

```python
# gate training: the cache was made without leaking slow features into fast inputs
state = fast_adapter(visual_tokens, question_tokens, video_meta)
gain = gate.predict_gain(state, cost_table["global_4"])
loss = ranking_or_margin_loss(gain, cached_net_gain)

# serving
state = fast_model.probe(batch)                 # no VGGT object in this process
decision = gate(state, budget)
answers = fast_model.generate(batch[~decision.seek])
slow_answers = slow_model.generate(batch[decision.seek])  # VGGT only here
return restore_original_order(answers, slow_answers)
```

慢路径调用前，frame IDs 必须与反事实 cache 和评估基线一致。调用后才允许使用 VGGT 可靠度做融合诊断；它不能改变该次是否调用的原始 gate 决定。

## 7. 必做测试与日志

| 测试 | 断言 |
|---|---|
| `test_no_geometry_forward` | `geometry_encoder_inputs=None` 能完成 image/video forward，且没有 geometry module。 |
| `test_self_no_vggt_call` | mock `VGGTEncoder.encode`；SELF 分支调用次数为 0。 |
| `test_seek_only_subbatch` | 混合 batch 中 VGGT 输入 batch size 等于 `seek.sum()`，输出顺序可恢复。 |
| `test_grid_alignment` | 输入的帧 ID、二维网格、mask 与 merger 输出一一对应；禁止 flat sequence 插值。 |
| `test_gate_no_leakage` | 删除 VGGT cache 字段后 gate 输入/输出不变；训练 schema 拒绝 external feature 字段。 |
| `test_checkpoint_contents` | `dualgeo-fast` state dict 中不含 `geometry_encoder.` 权重。 |

每个样本记录：`sample_id`、route、frame IDs、fast risk、predicted gain/cost、VGGT 调用次数、慢路径延迟/显存、最终答案、是否正确。训练运行还应记录数据 split、teacher/checkpoint hash、硬件与成本表版本。

## 8. 首轮命令与交付物

在代码实现前，先在 GeoThinker 分支中提供以下可重复入口（名称可调整，但语义保持）：

```bash
# M0：两条固定基线并导出反事实记录
python -m qwen_vl.dualgeo.collect_counterfactuals --route self --frames 4 ...
python -m qwen_vl.dualgeo.collect_counterfactuals --route seek --frames 4 ...

# M1：快系统；导出不含 VGGT 的 checkpoint
python -m qwen_vl.train.train_dualgeo --dualgeo_mode fast ...

# M3：训练 gate，再评测控制器
python -m qwen_vl.train.train_dualgeo --dualgeo_mode gate --counterfactual_cache ...
python -m qwen_vl.dualgeo.evaluate --policy learned_gate ...
```

第一个代码 PR 的完成标准不是取得新 SOTA，而是交付：两条真实分离的推理路径、可解析的反事实 cache、全部第 7 节测试、以及 always-SELF / always-SEEK / learned-gate 三组的准确率—延迟表。
