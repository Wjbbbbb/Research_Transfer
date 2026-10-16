# DualGeo M0：开发状态与接力手册

> 最后更新：2026-10-16
> 实现仓库：`/share/wangjingbin-local/projects/GeoThinker`
> 设计仓库：`/share/wangjingbin-local/projects/DualGeo`
> 工作分支：`dualgeo-m0`（基于 `e9b3786`）
> M0 bootstrap：`01f8e2d`（`feat(dualgeo): bootstrap M0 route separation`）

本文件是 DualGeo 的实现状态、Git 约定和下一会话入口。开发与实验必须先更新本文件，再开始会改变接口、数据 schema 或实验口径的工作。

## 1. 当前目标与边界

M0 只证明两条路径可以被真实拆开并能留下可复现的反事实记录：

- **SELF**：Qwen3-VL 主干不包含、加载或调用 `geometry_encoder` / VGGT。
- **SEEK**：复用已有 GeoThinker checkpoint、冻结 VGGT、现有 merger 和 cross-attention。
- 两路径对同一 `sample_id` 使用相同且有序的 4 个全图 `frame_ids`；记录答案、loss、正确性、端到端延迟、峰值显存。

M0 不训练 `FastSpatialAdapter`、不训练 gate、不做 ROI/tube/crop，也不声称任何新的任务性能。设计细节见 [`../../DualGeo/README.md`](../../DualGeo/README.md) 与 [`../../DualGeo/DEVELOPMENT.md`](../../DualGeo/DEVELOPMENT.md)。

## 2. 已完成

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

## 3. 尚未完成（M0 验收）

- [ ] 用 `geothinker` conda 环境运行单元测试。
- [x] 在一条真实图片样本上运行 `load_self_model`，确认 `assert_self_model_has_no_vggt` 成功且前向不访问 VGGT。
- [ ] 在同一输入上运行 SEEK，确认已有 checkpoint 的 VGGT 融合前向可用（SEEK 需要额外准备 `geometry_encoder_inputs`，不能复用 SELF 的标准 processor 输入）。
- [ ] 生成至少一对同帧 `SELF/SEEK` JSONL 记录，保存 cost 表与运行 manifest。
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
git add docs/dualgeo_m0_development.md src/qwen_vl/dualgeo tests/dualgeo \
  src/qwen_vl/model/qwenvl3/modeling_qwen3_vl.py
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
3. 将固定图片/视频样本和 4 帧规则写成可调用 collector，生成一对 JSONL records 及 manifest。
4. 只有 M0 验收全部通过且 commit 已记录后，才创建 M1 分支实现 `FastSpatialAdapter`。

## 8. 交接检查清单

交接时在消息中提供：当前 Git branch/commit、`git status --short`、本文件第 3 节勾选状态、最后一条测试命令和输出、最近 manifest 路径、任何 blocked 的外部依赖。不要只报“代码已完成”。
