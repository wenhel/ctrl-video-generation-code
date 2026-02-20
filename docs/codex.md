# Codex 分析任务：GenCompositor

本文档用于交给 Codex 做代码库分析与审查。分析范围为本仓库 GenCompositor（ICLR 2026，生成式视频抠图与合成）。

---

## 参考与范围

- **当前仓库**: `/home/wenhel/CODE/GenCompositor`
- **论文**: GenCompositor: Generative Video Compositing with Diffusion Transformer (ICLR 2026)  在 docs/2509.02460v1.pdf
- **核心依赖**: 基于 diffusers（本仓库内修改版）、CogVideoX-5B-I2V、SAM2 前景分割
- **建议分析范围**:
  - **必看**: `infer/`（推理脚本）、`train/`（训练脚本）、`utils/`、`app/`（Gradio 演示）
  - **选看**: `diffusers/` 下与 CogVideoX / 本任务相关的修改（如 pipeline、scheduler、model）
  - **选看**: `sam2/` 仅作调用关系与接口理解，可不深入内部实现

---

## 项目简要

- **任务**: 生成式视频合成——将前景视频的身份与运动注入到背景视频，支持轨迹、尺寸等控制。
- **流程概览**:
  - 推理: 背景/前景视频 → 预处理（`preprocess_bg_fg_videos.sh`）→ SAM2 前景 mask（`get_fgmask.py`）→ 运动轨迹/遮罩（`get_movemask.py`、`usr.py`）→ 合成（`testinput.py`，调用 CogVideoX 相关 pipeline）。
  - 训练: VideoComp 数据集（filtered_mask、filtered_masked_video、inpainted_sum、GTs、fg）→ `train_cogvideox_compositing_sep.py`（含 DiT、branch、ERoPE 等）。
- **关键组件**: DiT 背景保留分支（masked token injection）、DiT 融合块（full self-attention）、ERoPE、前景增强与数据管线。

---

## 任务描述

1. **代码结构梳理**  
   按模块（infer / train / utils / app，以及 diffusers 中与本项目直接相关的部分）梳理：每个主要 Python 文件的职责，以及每个文件内重要 class/function 的一句话说明。

2. **数据流与调用链**  
   从「输入（背景/前景视频、轨迹、mask）」到「最终合成视频」的完整数据流；训练时从 VideoComp 目录到 loss 的数据流。标出关键函数、中间表示（tensor 形状/含义可简要注明）。

3. **与论文/README 的对应**  
   对照 README 与论文中的方法描述，标出：背景保留分支、前景融合块、ERoPE、前景增强等在代码中的落点（文件+类/函数）。

4. **潜在问题与改进点**  
   包括但不限于：  
   - 可能的 bug 或边界情况（如分辨率、帧数、路径、设备）。  
   - 代码重复、配置硬编码、可维护性。  
   - 与 diffusers 上游或 CogVideoX 的接口变更风险。  
   - 训练/推理在数据预处理或模型接口上的一致性。  
   每个点请简要说明位置与建议。

5. **可选的扩展与复用**  
   若要在本仓库上做「新前景/新轨迹/新分辨率」或「接入其他 backbone」等扩展，当前结构下需要改动的入口与最小改动路径。

---

## 结果要求

请输出以下文档（可放在 `docs/summray/` 下，或按你的习惯命名）：

1. **summary-codebase.md**  
   代码库结构总览：各模块职责、每个主要 py 的一句话说明、每个 py 内重要 class/function 的一句话说明（可表格或列表）。

2. **summary-dataflow.md**  
   推理与训练两条数据流：从输入到输出的步骤、关键函数与张量形态/含义、与配置/脚本的对应关系。

3. **summary-paper-to-code.md**  
   论文/README 中的方法点与代码落点对照（背景分支、融合块、ERoPE、数据增强等）。

4. **summary-issues-and-suggestions.md**  
   潜在问题与改进建议列表；每项标注优先级（如 H1/M1/L1）并注明涉及文件/行或模块。

5. **summary-extension-guide.md**（可选）  
   基于当前代码结构的扩展指南：新场景/新模型/新数据时建议改动的入口与步骤。

---

## 使用方式

- 若使用 **codex-runner.sh**：将本文件路径作为 task 传入，workspace 设为仓库根目录 `/home/wenhel/CODE/GenCompositor`，输出会写入指定 log。
- 若使用 **consult_codex**：将本文件内容或要点作为 prompt，directory 设为 `/home/wenhel/CODE/GenCompositor`，让 Codex 按上述任务与结果要求进行分析。
