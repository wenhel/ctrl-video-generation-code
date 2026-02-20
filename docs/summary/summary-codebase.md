# GenCompositor 代码库结构总览

> ICLR 2026 - Generative Video Compositing with Diffusion Transformer

---

## 顶层目录结构

```
GenCompositor/
├── app/            # Gradio Web 演示界面
├── assets/         # 示例视频与图片素材
├── diffusers/      # 修改版 diffusers 库（核心模型与管线）
├── docs/           # 文档
├── infer/          # 推理脚本
├── sam2/           # SAM2 前景分割（第三方，调用接口）
├── train/          # 训练脚本
├── utils/          # 通用工具函数
├── VideoComp_demo/ # 训练数据集示例
├── env.sh          # 环境变量设置
├── requirements.txt
└── README.md
```

---

## 1. infer/ — 推理模块（5 个 Python 文件）

| 文件 | 职责 |
|------|------|
| `get_bgvideo.py` | 背景视频预处理：缩放到 720×480、帧采样到 49 帧、ping-pong 补帧 |
| `get_fgmask.py` | 前景分割：用 Grounding DINO 检测 + SAM2 分割，提取前景 mask 与元素视频 |
| `get_movemask.py` | 轨迹绘制 + mask 视频生成：交互式绘制运动轨迹，将前景贴入轨迹位置生成 mask |
| `usr.py` | 用户轨迹绘制工具：独立的交互式 GUI，支持光流追踪 / 线性插值 / 采样 |
| `testinput.py` | **主推理管线**：加载 CogVideoX 模型，执行 inpainting 合成，输出最终视频 |

### testinput.py 重要函数

| 函数 | 说明 |
|------|------|
| `generate_video(...)` | 主入口，加载模型、读取数据、调用 pipeline、保存结果（~50 个参数） |
| `read_video_with_mask(...)` | 加载背景/mask/前景视频，执行形态学清洗、高斯平滑、连通域过滤 |
| `quick_freeze(model)` | 冻结模型参数用于推理 |
| `get_gaussian_kernel(...)` | 生成高斯模糊卷积核用于 mask 边缘平滑 |
| `_visualize_video(...)` | 拼接原始/mask/前景/合成视频用于可视化 |

### get_fgmask.py 重要函数

| 函数 | 说明 |
|------|------|
| `create_fg_video(...)` | 从 mask + 原始帧创建前景元素视频（576×576 白色画布） |
| 主流程 | Grounding DINO 检测 → SAM2 图像 mask → SAM2 视频传播 → 保存 per-frame mask |

### get_movemask.py / usr.py 轨迹模式

| 模式 | 触发条件 | 方法 |
|------|---------|------|
| 光流追踪 | 单点点击 | Lucas-Kanade 光流 |
| 线性插值 | 点数 < 帧数 | 按段均匀插值 |
| 均匀采样 | 点数 > 帧数 | 下采样轨迹点 |

---

## 2. train/ — 训练模块（2 个核心文件）

| 文件 | 职责 |
|------|------|
| `train_cogvideox_compositing_sep.py` | **主训练脚本**（2275 行），包含数据集、collate、训练循环、验证 |
| `mask_process.py` | Mask 变换增强：brush / random_brush / rect / ellipse / circle 五种模式 |
| `train.sh` | 训练启动脚本，`accelerate launch` + DeepSpeed |
| `accelerate_config_machine_single_ds.yaml` | 分布式配置：8 GPU、ZeRO-2、bf16 |
| `train_demo.csv` / `val_demo.csv` | 数据元信息（path, start_frame, end_frame, fps, caption） |

### train_cogvideox_compositing_sep.py 重要类/函数

| 类/函数 | 说明 |
|---------|------|
| `VideoInpaintingDataset` | 数据集类：从 CSV 加载 GT/前景/masked 视频和 mask |
| `MyWebDataset` | Collate 函数：prompt 分词、视频裁剪/归一化、mask 变换 |
| `main()` | 训练主循环：模型加载 → 数据准备 → 扩散训练 → checkpoint & 验证 |
| `log_validation()` | 验证函数：加载推理 pipeline、生成样本、上传 wandb |
| `get_args()` | 80+ 训练参数配置 |
| `prepare_rotary_positional_embeddings()` | 准备 3D 旋转位置编码 |

### 训练模型组件

| 组件 | 参数量 | 状态 |
|------|--------|------|
| T5 Text Encoder | 4.76B | 冻结 |
| CogVideoXTransformer3D3BModel_sep | 5.57B | **可训练** |
| AutoencoderKLCogVideoX (VAE) | 215M | 冻结 |
| CogvideoXBranchModel | 301M | 冻结 |

### 训练核心超参

| 参数 | 值 |
|------|-----|
| 分辨率 | 480×720 |
| 帧数 | 49 |
| FPS | 8 |
| batch size | 1 / GPU |
| 学习率 | 2e-5 |
| inpainting loss 权重 | 5.0 |
| mask 变换概率 | 0.3 |
| checkpoint 间隔 | 1024 steps |
| 验证间隔 | 256 steps |

---

## 3. utils/ — 工具模块（5 个文件）

| 文件 | 职责 |
|------|------|
| `video_utils.py` | 前景视频处理：帧二值化、形态学开运算、自适应缩放、白色画布合成 |
| `mask_dictionary_model.py` | Mask 数据模型：`MaskDictionaryModel` 管理帧级 mask 与目标元数据，IoU 匹配 |
| `common_utils.py` | 可视化工具：`CommonUtils` 在图像上绘制 mask 与 bbox |
| `track_utils.py` | 追踪辅助：`sample_points_from_masks()` 从 mask 中采样点用于 SAM2 |
| `supervision_utils.py` | 16 色颜色映射表，用于一致的可视化着色 |

---

## 4. app/ — Gradio 演示（1 个文件）

| 文件 | 职责 |
|------|------|
| `app.py` | 完整的 Web 演示界面，集成前景分割、轨迹绘制、mask 生成、视频合成全流程 |

### app.py 三步流程

| 步骤 | 功能 | 关键函数 |
|------|------|---------|
| Step 1 | 前景处理 | `process_foreground_video()` — SAM2 分割前景 |
| Step 2 | 背景与轨迹 | `draw_and_save_trajectory()` — 交互式轨迹 |
| Step 3 | 视频合成 | `generate_video()` — CogVideoX inpainting |

---

## 5. diffusers/ — 修改版 Diffusers（核心模型层）

### 5.1 Pipeline（管线）

| 文件 | 类名 | 说明 |
|------|------|------|
| `pipeline_cogvideox_inpainting_i2v_branch3_sep.py` | `CogVideoXI2VTriInpaintPipeline_sep` | **[核心]** 三分支 I2V inpainting，前景/背景分离处理 |
| `pipeline_cogvideox_inpainting_branch.py` | `CogVideoXDualInpaintPipeline` | 双分支 inpainting |
| `pipeline_cogvideox_inpainting_i2v_branch_anyl.py` | `CogVideoXI2VDualInpaintAnyLPipeline` | 灵活的双分支 I2V |
| `pipeline_cogvideox_inpainting_sft.py` | `CogVideoXSFTInpaintPipeline` | SFT 微调 inpainting |
| `pipeline_cogvideox_inpainting_selfguidance.py` | `CogVideoXSelfGuidanceInpaintPipeline` | 自引导 inpainting |
| `pipeline_cogvideox.py` | `CogVideoXPipeline` | 基础视频生成（原版） |
| `pipeline_cogvideox_image2video.py` | `CogVideoXImageToVideoPipeline` | I2V 基础版（原版） |
| `pipeline_cogvideox_inpainting.py` | `CogVideoXInpaintPipeline` | 基础 inpainting（原版） |

### 5.2 模型（Transformer & Branch）

| 文件 | 类名 | 说明 |
|------|------|------|
| `cogvideox_transformer_3d_fg3b_sep.py` | `CogVideoXTransformer3D3BModel_sep` | **[核心]** 独立 patch embedding 的三分支 Transformer |
| `cogvideox_transformer_3d_fg3b.py` | `CogVideoXTransformer3D3BModel` | 三分支 Transformer（融合版） |
| `branch_cogvideox.py` | `CogvideoXBranchModel` | **[核心]** 轻量级背景保留分支（30 层，zero-init） |
| `cogvideox_transformer_3d.py` | `CogVideoXTransformer3DModel` | 标准 CogVideoX DiT（原版） |

### 5.3 Embedding

| 类/函数 | 文件 | 说明 |
|---------|------|------|
| `CogVideoXPatchEmbed_sep` | `embeddings.py` | **[核心]** 分离的前景/背景 patch embedding（双投影头） |
| `get_3d_rotary_pos_embed()` | `embeddings.py` | **[核心]** ERoPE — 3D 旋转位置编码 |
| `CogVideoXPatchEmbed` | `embeddings.py` | 标准 patch embedding（原版） |

### 5.4 注意力

| 类 | 说明 |
|----|------|
| `CogVideoXAttnProcessor2_0` | 标准 scaled dot-product attention + RoPE |
| `CogVideoXAttnProcessor2_0_wo_text` | 无文本注意力（背景分支使用） |
| `CogVideoXAttnProcessor2_0_resample` | 重采样注意力 |

### 5.5 调度器

| 类 | 说明 |
|----|------|
| `CogVideoXDPMScheduler` | DPM-Solver 调度器（CogVideoX 适配） |
| `CogVideoXDDIMScheduler` | DDIM 调度器（CogVideoX 适配） |

---

## 6. 外部依赖

| 模块 | 用途 | 调用方 |
|------|------|--------|
| SAM2 (sam2/) | 视频前景分割 | `get_fgmask.py`, `app.py` |
| Grounding DINO | 零样本目标检测（文本→bbox） | `get_fgmask.py`, `app.py` |
| CogVideoX-5b-I2V | 基础视频扩散模型 | 推理 & 训练 |
| T5 Encoder | 文本编码 | 训练 |
| DeepSpeed | 分布式训练（ZeRO-2） | 训练 |
| wandb | 训练可视化 | 训练 |
