# GenCompositor 潜在问题与改进建议

> 优先级: H = 高, M = 中, L = 低

---

## 1. Bug 与边界情况

### H1: GPU 设备硬编码
- **位置**: `infer/get_fgmask.py` — `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`
- **问题**: 硬编码 GPU 1，在单 GPU 或多用户环境下会失败
- **建议**: 改为 CLI 参数或环境变量传入

### H2: 帧数假设固定为 49
- **位置**: 全局（`get_bgvideo.py`, `get_fgmask.py`, `testinput.py`, 训练脚本等）
- **问题**: 49 帧是 CogVideoX 的特定约束，多处硬编码。如果未来换模型或需要不同帧数，改动面大
- **建议**: 提取为全局常量 `MAX_FRAMES = 49`，集中管理

### M1: 前景视频尺寸检查缺失
- **位置**: `infer/testinput.py::read_video_with_mask()`
- **问题**: 前景视频缩放到背景尺寸时，如果前景宽高比与背景差异过大，居中 padding 可能导致前景被严重压缩
- **建议**: 添加宽高比检查和警告

### M2: Mask 连通域过滤阈值硬编码
- **位置**: `infer/testinput.py` — `area_threshold = 0.0001`（0.01%）
- **问题**: 不同分辨率下，固定比例阈值可能过于激进或保守
- **建议**: 根据实际分辨率自适应调整，或作为参数暴露

### M3: 光流追踪鲁棒性
- **位置**: `infer/get_movemask.py`, `infer/usr.py` — Lucas-Kanade 光流
- **问题**: 单点 LK 光流在快速运动、遮挡、大位移场景容易丢失
- **建议**: 添加追踪质量检查（如 forward-backward 一致性检查）

### L1: ffmpeg 解码失败无优雅处理
- **位置**: `train/train_cogvideox_compositing_sep.py::VideoInpaintingDataset.__getitem__()`
- **问题**: ffmpeg 解码出错时直接 crash，没有 fallback
- **建议**: 添加异常捕获，返回 batch 中下一个有效样本

---

## 2. 代码重复与可维护性

### H3: 函数大量重复
- **位置**:
  - `read_video_with_mask()` — 在 `testinput.py` 和 `app.py` 中各有一份
  - `generate_video()` — 在 `testinput.py` 和 `app.py` 中各有一份
  - `draw_and_save_trajectory()` — 在 `get_movemask.py` 和 `usr.py` 中各有一份
  - `get_gaussian_kernel()` — 在 `testinput.py` 和 `app.py` 中各有一份
  - `quick_freeze()` — 在 `testinput.py` 和 `app.py` 中各有一份
  - `create_fg_video()` — 在 `get_fgmask.py` 和 `utils/video_utils.py` 中各有一份
- **问题**: 修改一处时容易遗漏另一处，导致行为不一致
- **建议**: 提取公共函数到 `utils/` 模块，所有文件统一导入

### M4: testinput.py 过大
- **位置**: `infer/testinput.py` — 21KB, ~50 个 CLI 参数
- **问题**: 数据加载、模型加载、推理逻辑全在一个文件中，难以维护
- **建议**: 拆分为 `model_loader.py`, `data_processor.py`, `inference.py`

### M5: train 脚本过大
- **位置**: `train/train_cogvideox_compositing_sep.py` — 2275 行
- **问题**: 数据集、collate、训练循环、验证、工具函数全在一个文件
- **建议**: 至少拆分 dataset 和 collate 到独立文件

---

## 3. 配置与硬编码

### H4: 模型路径硬编码
- **位置**: `app/app.py` — `/path/to/checkpoint-45056` 等
- **问题**: 部署时必须手动修改代码
- **建议**: 使用配置文件（YAML/JSON）或环境变量管理所有路径

### M6: WANDB API Key 硬编码位置
- **位置**: `train/train_cogvideox_compositing_sep.py:81`
- **问题**: 需要直接修改代码填入 API Key
- **建议**: 从环境变量 `WANDB_API_KEY` 读取（wandb 原生支持）

### M7: 分辨率/FPS 分散定义
- **位置**: 各文件独立定义 `720×480`, `576×576`, `fps=12`, `fps=8` 等
- **问题**: 值分散在多个文件中，不一致风险
- **建议**: 创建 `config.py` 集中管理所有默认参数

### L2: Gaussian kernel 参数硬编码
- **位置**: `testinput.py` — `kernel_size=51, sigma=10`; morphology kernel `3×3`
- **问题**: 不同场景可能需要不同参数
- **建议**: 作为可选 CLI 参数暴露

---

## 4. 接口变更风险

### H5: 自定义 diffusers 与上游不兼容
- **位置**: 整个 `diffusers/` 目录
- **问题**: 本仓库包含完整的 diffusers 库副本，且做了大量自定义修改。上游 diffusers 更新后无法直接升级
- **建议**:
  - 记录所有修改点（已在本文档 summary-paper-to-code.md 中部分覆盖）
  - 考虑仅 fork 修改的文件，通过 monkey-patch 或继承方式集成
  - 维护 diff 记录，便于未来 rebase

### M8: CogVideoX 接口变更
- **位置**: `diffusers/.../pipelines/cogvideo/` 下所有自定义 pipeline
- **问题**: 如果 CogVideoX 上游修改了模型接口（如输入通道数、位置编码维度），需要同步更新
- **建议**: 锁定 CogVideoX 版本，记录依赖的具体 commit

### M9: SAM2 / Grounding DINO 版本锁定
- **位置**: `infer/get_fgmask.py`, `app/app.py`
- **问题**: SAM2 和 Grounding DINO 均在快速迭代，API 可能变化
- **建议**: 在 `requirements.txt` 中锁定精确版本

---

## 5. 训练/推理一致性

### H6: Mask 处理不一致
- **位置**:
  - 推理: `testinput.py::read_video_with_mask()` — 腐蚀+连通域+膨胀+高斯
  - 训练: `mask_process.py::transform_video_masks()` — 5 种变换模式
- **问题**: 推理时的 mask 处理逻辑与训练时不同。训练时有多种增强，但推理时只有固定处理
- **建议**: 确保推理时的 mask 处理是训练时某一模式的子集；或在推理时也支持选择模式

### M10: FPS 不一致
- **位置**:
  - `get_bgvideo.py` — 输出 12fps
  - `testinput.py` — 输出 8fps
  - 训练 — 默认 8fps
- **问题**: 背景预处理用 12fps 但最终输出和训练都用 8fps
- **建议**: 统一 FPS 标准，或明确文档说明 FPS 转换逻辑

### L3: Gamma 校正未实际使用
- **位置**: `testinput.py::apply_consistent_gamma()` — 定义但未在主流程中调用
- **问题**: 死代码，增加理解成本
- **建议**: 移除或标记为实验性功能

---

## 6. 性能与资源

### M11: VAE 编码重复计算
- **位置**: `train/train_cogvideox_compositing_sep.py` — 训练循环
- **问题**: 每个 batch 对 4 个视频流分别做 VAE 编码，但 VAE 冻结不变。可预计算 latent 缓存
- **建议**: 支持 offline latent 预计算模式，显著加速训练

### M12: 前景处理串行化
- **位置**: `infer/get_fgmask.py::create_fg_video()` — 前 12 帧并行，其余串行
- **问题**: 为什么只并行前 12 帧？可能是遗留代码
- **建议**: 全部帧并行处理或使用更高效的批处理

### L4: Branch 模型每步都重算
- **位置**: Pipeline 推理循环
- **问题**: Branch 输入在所有去噪步骤中不变，但每步都重新计算
- **建议**: Branch 结果可缓存，仅计算一次（需验证 timestep 是否影响 branch 输出）

---

## 总结优先级

| 优先级 | 编号 | 问题 |
|--------|------|------|
| **高** | H1 | GPU 硬编码 |
| **高** | H2 | 帧数 49 硬编码 |
| **高** | H3 | 大量函数重复 |
| **高** | H4 | 模型路径硬编码 |
| **高** | H5 | diffusers 上游不兼容 |
| **高** | H6 | 训练/推理 mask 处理不一致 |
| **中** | M1-M12 | 各类中等优先级问题 |
| **低** | L1-L4 | 低优先级改进 |
