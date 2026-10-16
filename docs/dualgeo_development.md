# DualGeo：开发状态与接力手册

> 最后更新：2026-10-16
> 唯一仓库：`/share/wangjingbin-local/projects/DualGeo`
> 当前分支：`dualgeo-m1-fast`（基于 M0 验收 `222bde8`）
> M0 bootstrap：`01f8e2d`（`feat(dualgeo): bootstrap M0 route separation`）
> M0 验收增量：`dde0d65`（`feat(dualgeo): complete M0 counterfactual collection`）

本文件是 DualGeo 的实现状态、Git 约定和下一会话入口。开发与实验必须先更新本文件，再开始会改变接口、数据 schema 或实验口径的工作。

## 1. 当前目标与边界

本文跨阶段记录 DualGeo 的实现状态、实验口径和 Git 接力规则。M0 已结束；M1 只构建并验证快系统能力与 curriculum 对照，不训练 gate、不改变慢路径融合。

M0 的目标是证明两条路径可以被真实拆开并能留下可复现的反事实记录：

- **SELF**：Qwen3-VL 主干不包含、加载或调用 `geometry_encoder` / VGGT。
- **SEEK**：复用已有 GeoThinker checkpoint、冻结 VGGT、现有 merger 和 cross-attention。
- 两路径对同一 `sample_id` 使用相同且有序的 4 个全图 `frame_ids`；记录答案、loss、正确性、端到端延迟、峰值显存。

M0 不训练 `FastSpatialAdapter`、不训练 gate、不做 ROI/tube/crop，也不声称任何新的任务性能。设计细节见 [研究设计](dualgeo_design.md) 与 [开发蓝图](dualgeo_architecture.md)。

## 1.1 M1 当前目标：快系统与训练顺序对照

M1 的快系统仍是同一 Qwen3-VL 的 `SELF` 路径：它不加载 VGGT；`FastSpatialAdapter` 只读取 SELF Qwen token，输出空间摘要、辅助 predicate logits 和未校准风险 logits。答案仍由 SELF Qwen 生成，不能把 adapter 单独称作第二个 VLM。

要比较的两条路线必须从**相同的原始** `/share/wangjingbin-local/models/Qwen3-VL-8B-Instruct` 初始化，并固定训练样本、token 预算、可训练参数、随机种子和最终联合巩固预算：

```text
A: SELF-only → SEEK-only → SELF/SEEK 联合巩固
B: SEEK-only → SELF-only → SELF/SEEK 联合巩固
```

联合巩固是强制阶段；否则第二阶段主要测到的是对第一阶段的遗忘，无法比较可部署的双系统。已有 GeoThinker checkpoint 只能作为 always-SEEK 参考，不能作为 A/B 的共同起点。

M1 已完成：

- [x] 从 `222bde8` 建立分支 `dualgeo-m1-fast`。
- [x] 新增纯 SELF token 模块：`FastSpatialAdapter`、风险与可选 predicate head；模块中不含 geometry/VGGT state。
- [x] 新增 ECE、Brier、risk-coverage 指标与两条训练顺序的不可变契约。
- [x] 在 `geothinker` 环境中运行 M0+M1 单元测试（10/10 通过）。

`models/train/mindcube_10k.json` 是本机第一候选训练资产；其分组切分已写入本地忽略目录 `artifacts/dualgeo/m1/mindcube-split-seed0/`：seed `0`、源文件 SHA256 `600d857ff1ab…`，train/calibration/test 分别为 `7875/1107/1018` 个问答、`575/72/79` 个帧目录组。不能直接随机混合其问答到各 split，避免同场景泄漏。

M1 的可执行 curriculum 入口已建立，但正式训练尚未启动：必须先完成本轮纯 Qwen VSI-Bench 基线。入口为 `scripts/dualgeo/train_m1_curriculum.sh`，它从原始 `/share/wangjingbin-local/models/Qwen3-VL-8B-Instruct` 开始，接受 `self_then_seek` 或 `seek_then_self`，顺序运行两个单路阶段和一个联合巩固阶段。每一阶段固定同一 split、seed、全局 batch/token 步数和 Qwen 可训练参数。输出必须在 `artifacts/dualgeo/m1/`，不能进入 Git。

为使比较公平，SEEK 的 `/share/wangjingbin-local/models/VGGT-1B`、geometry merger 与各层 cross-attention 均是**冻结 sidecar**；后两者由现有 `models/GeoThinker-VGGT-Scaled-Qwen3VL-8B` 只读加载。它们不是 A/B 的 Qwen 初始权重：两条路线的可训练 Qwen 均从原始 Qwen3-VL 开始。`balanced_hash` 联合阶段以 `sample_id + seed` 的稳定哈希为每个 batch-size-1 样本选择 SELF 或 SEEK，因而可恢复、不会读取答案或慢路径输出，也不训练任何 gate。

运行前必须完成以下 preflight：在空闲的两张 GPU 上以 `NPROC_PER_NODE=2`、`TOTAL_BATCH_SIZE=8` 跑每个顺序的 `max_steps=2`；验证 SELF→SEEK 和 SEEK→SELF checkpoint 转换、冻结清单、loss、最终 save、以及 joint 阶段确实同时出现无 `geometry_encoder_inputs` 与有该字段的 batch。通过后才可将同一脚本的第三个参数改为正式固定步数；不得把 preflight checkpoint 用作正式初始化。

## 2. M0 已完成

- [x] 建立 Git 分支 `dualgeo-m0`。
- [x] 移除 GeoThinker 中新增的 3DThinker 实现：`three_d_thinker.py`、trainer、训练脚本和复现说明；同步移除了参数、collator latent 插入、Qwen3 latent branch 与 trainer dispatch。
- [x] 新建 `src/qwen_vl/dualgeo/`：
  - `types.py`：只含 `SELF/SEEK` 和严格 `CounterfactualRecord` schema；拒绝 VGGT feature 等未知字段。
  - `cache.py`：JSONL 读写与 schema 验证。
  - `profiling.py`：CUDA 前后同步、峰值显存和端到端耗时。
  - `loading.py`：显式生成无几何 encoder 的 SELF config，并分别加载 SELF/SEEK 模型。
- [x] 修复 `Qwen3VLModel.forward` 对 `geometry_encoder_inputs[0]` 的隐式依赖：无 geometry input 时从 image/video grid 推断视觉输入数，允许 custom GeoThinker 模型完成无几何前向。
- [x] 增加 `tests/dualgeo/test_m0_primitives.py`：cache round-trip、泄漏字段拒绝、CPU 计时。
- [x] `python -m compileall` 通过（仅语法检查）。
- [x] 在 `geothinker` 环境运行 M0 单元测试（3/3 通过），验证 SELF config 会关闭 geometry encoder/cross-attention。
- [x] 真实 checkpoint smoke：SELF 在 `cuda:0` 加载后 `geometry_module_count=0`；SEEK 在 `cuda:1` 加载后 `geometry_module_count=1381`。二者均为 BF16。
- [x] 真实 SELF multimodal forward：使用 MindCube 本地图片、未传 `geometry_encoder_inputs`，得到 logits shape `[1, 446, 151936]`，证明 M0 的空几何输入修复有效。
- [x] 固定四帧输入协议：`inputs.py` 要求且仅接受 4 个有序全图帧；Qwen 和 VGGT 输入从同一有序路径列表构造。
- [x] 真实四帧前向：同一 MindCube 四帧在 `cuda:1` 上均得到 logits `[1, 152, 151936]`；SELF 的 `geometry_module_count=0`，SEEK 为 `1381`。
- [x] 成对 collector：`python -m qwen_vl.dualgeo.collect` 会在独立模型生命周期中依次运行 SELF、SEEK，生成严格 schema JSONL 与不可入训的 JSON manifest。
- [x] 端到端采集 smoke：`artifacts/dualgeo/m0-smoke-device/`（本地忽略）包含同一四帧的两个记录；SELF/SEEK 均生成 `ok`，teacher-forced loss 分别为 `0.00662/0.00393`，峰值显存为 `16775.59/22941.76 MiB`。此人为短答仅验证采集链路，**不是**空间能力结果。

## 3. M0 验收清单（已完成）

- [x] 用 `geothinker` conda 环境运行单元测试（本轮 6/6 通过）。
- [x] 在一条真实图片样本上运行 `load_self_model`，确认 `assert_self_model_has_no_vggt` 成功且前向不访问 VGGT。
- [x] 在同一输入上运行 SEEK，确认已有 checkpoint 的 VGGT 融合前向可用（SEEK 使用额外准备的 `geometry_encoder_inputs`）。
- [x] 生成至少一对同帧 `SELF/SEEK` JSONL 记录，并保存 cost 表与运行 manifest。
- [x] M0 bootstrap 已提交为 `01f8e2d`；提交内容经 staged diff 审计，未混入既有脏工作区改动。

## 4. 已知资产与路径

| 资产 | 路径/值 | 用途 |
|---|---|---|
| SEEK VLM checkpoint | `models/GeoThinker-VGGT-Scaled-Qwen3VL-8B` | 已有 Qwen3-VL + GeoThinker 融合权重。 |
| VGGT 权重 | 内嵌于上述 GeoThinker checkpoint | M0 的 SEEK loader 默认直接读取 checkpoint 内权重；仅在需要覆盖时传 `geometry_encoder_path=facebook/VGGT-1B`。 |
| conda 环境 | `geothinker` | 所有真实加载、GPU profile 与 eval 使用它。 |
| SELF checkpoint 来源 | 上述 SEEK checkpoint 的 Qwen 权重 | `load_self_model` 将 `use_geometry_encoder=False`、`geo_cross_attn=False` 后加载；几何权重应作为未使用权重被忽略。 |

`models/`、`data/media/`、`logs/`、`.hf_cache/` 是本地资产，均不得加入 Git。运行产物应写入 `artifacts/dualgeo/`（已在 `.gitignore` 中忽略后再使用）。

## 5. Git 工作流（必须遵守）

1. **每个阶段一个分支。**M0 使用 `dualgeo-m0`；M1 从 M0 commit 新建 `dualgeo-m1-fast`，不要直接在 `main` 开发。
2. **一次提交只含一个可验证目的。**推荐顺序：`chore: remove 3dthinker branch`、`feat(dualgeo): add m0 route primitives`、`test(dualgeo): ...`。本轮可在检查后合并为一个 M0 bootstrap commit，但不得把未关联的已有改动带入。
3. **提交前必须审计。**依次运行 `git status --short`、`git diff --check`、目标测试，并以 `git diff --cached --stat` 确认 staged 文件仅属于 DualGeo/3DThinker 清理。
4. **不提交大资产和结果。**权重、视频、Hugging Face cache、日志和 JSONL 反事实数据只记录其路径、hash、命令、数据 split 与硬件信息；不进 Git。
5. **记录可复现性。**每次实验写 manifest：Git commit、dirty 状态、checkpoint/config hash、VGGT revision、processor、frame rule、数据 split、GPU/driver、命令、p50/p95 latency、峰值显存。
6. **不要覆盖既有脏改动。**本仓库在 M0 开始前已有与 DualGeo 无关的修改（评测、数据路径、sampler 等）。除非用户明确要求，保持 unstaged、不回滚、不纳入 DualGeo commit。

建议提交前命令：

```bash
git status --short
git diff --check
PYTHONPATH=src conda run -n geothinker python -m unittest discover -s tests/dualgeo -p 'test_*.py'
git add docs/dualgeo_m0_development.md src/qwen_vl/dualgeo tests/dualgeo
# 若 shared model 文件已有他人未提交改动，使用交互式分块暂存，绝不直接 git add 整个文件。
git add -p src/qwen_vl/model/qwenvl3/modeling_qwen3_vl.py
git diff --cached --stat
git commit -m "feat(dualgeo): bootstrap M0 route separation"
```

3DThinker 清理涉及现有的已修改文件；在 staged diff 中确认只删除其相关 hunks后再把这些删除一并提交。

## 6. M0 数据与 schema 约束

`CounterfactualRecord` 版本固定为 `dualgeo.m0.v1`。同一个 `sample_id` 必须有一条 SELF 和一条 SEEK 记录，且下列字段完全相同：`frame_ids`、数据 split、问题版本、评估规则。M0 record 包含：

```json
{
  "sample_id": "...",
  "route": "self | seek",
  "frame_ids": [0, 8, 16, 24],
  "answer": "...",
  "loss": 0.0,
  "correct": true,
  "profile": {"wall_time_ms": 0.0, "peak_memory_mb": 0.0},
  "checkpoint_id": "...",
  "schema_version": "dualgeo.m0.v1"
}
```

禁止保存或传入 gate 的字段：VGGT token、camera/pose、external reliability、slow feature、慢路径中间 hidden state。它们会让后续 gate 在部署时不可获得的信息上训练。

## 7. 下一会话的执行顺序

1. 确认当前分支和工作区，再运行 `PYTHONPATH=src conda run -n geothinker python -m unittest discover -s tests/dualgeo -p 'test_*.py'`。
2. 运行 `python -m qwen_vl.dualgeo.smoke` 对真实 checkpoint 做 load-only smoke test；先加载 SELF，再加载 SEEK，避免同时占满 GPU 显存。
3. 为 `models/train/mindcube_10k.json` 建立按原始场景/视频分组的 train/calibration/test manifest；记录算法、seed、source hash 和样本/场景计数，产物写入 `artifacts/dualgeo/`。
4. 为 A/B 两条路线建立同 token 预算、同可训练参数的阶段配置。必须从原始 Qwen3-VL 权重开始，而不是从 GeoThinker checkpoint 开始。
5. 运行纯 Qwen VSI-Bench 基线：使用 `geothinker` lmms-eval adapter、原始 Qwen 权重、32 帧、greedy、seed 0；完整 5,130 请求正在写入 `artifacts/dualgeo/m1/vsibench-qwen3-baseline-full/`。它不加载 VGGT。
6. 在开跑长训练前，以一小批 MindCube 样本验证 SELF-only、SEEK-only 与 joint 阶段的 checkpoint 转换、冻结策略、loss 与恢复路径；随后才提交正式训练任务。

## 8.1 已知非阻塞警告

- SELF load 会报告 checkpoint 中的 VGGT 和 cross-attention 权重未被使用。这符合 SELF 配置主动关闭 geometry encoder/cross-attention 的预期；`geometry_module_count=0` 是应检查的正向证据。
- SEEK load 会报告一部分未使用的 merger/cross-attention 权重。现有 GeoThinker eval 已使用该 checkpoint；M0 先记录该事实，后续 SEEK 真正前向时应确认这些权重差异是否来自注入层配置而非 checkpoint 错配。

## 9. 交接检查清单

交接时在消息中提供：当前 Git branch/commit、`git status --short`、本文件第 3 节勾选状态、最后一条测试命令和输出、最近 manifest 路径、任何 blocked 的外部依赖。不要只报“代码已完成”。
