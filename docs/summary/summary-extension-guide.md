# GenCompositor 扩展指南

---

## 1. 新前景/新场景接入

### 最小改动路径

只需准备新的输入视频，使用现有推理管线即可：

```bash
# 1. 准备背景视频
python infer/get_bgvideo.py --bg_video_path <新背景.mp4> --save_path <输出.mp4>

# 2. 前景分割（修改 text_prompt 为新目标物体）
python infer/get_fgmask.py \
    --fg_video_path <新前景.mp4> \
    --text_prompt "<目标物体名称>" \
    ...

# 3. 绘制轨迹
python infer/get_movemask.py \
    --fg_video_path <前景元素.mp4> \
    --video_path <背景.mp4> \
    --rescale <缩放比例> \
    ...

# 4. 合成
python infer/testinput.py \
    --fg_video_path <前景元素.mp4> \
    --video_path <背景.mp4> \
    --mask_path <mask视频.mp4> \
    --output_path <输出.mp4>
```

### 需要注意的点

- **前景分割失败**: 如果 Grounding DINO 无法检测目标，需手动提供 bbox 或换用其他检测模型
- **复杂前景**: 多目标、遮挡严重的场景可能需要多次 SAM2 分割
- **缩放比例**: `--rescale` 参数控制前景在背景中的相对大小，需根据场景调整

---

## 2. 新轨迹类型

### 当前支持

| 模式 | 触发 | 文件 |
|------|------|------|
| 手动绘制 | 鼠标拖拽 | `get_movemask.py`, `usr.py` |
| 光流追踪 | 单点点击 | 同上 |
| 线性插值 | 少量关键点 | 同上 |

### 扩展入口

**文件**: `infer/get_movemask.py::draw_and_save_trajectory()`

添加新轨迹类型的步骤：
1. 在 `draw_and_save_trajectory()` 中添加新的 case
2. 生成 `trajectory.txt`（CSV 格式，每行 `x,y`，行数 = 帧数）
3. `generate_mask_video_with_trajectory()` 不需修改，它只读取轨迹坐标

**示例 — 添加贝塞尔曲线轨迹**:
```python
# 在 draw_and_save_trajectory() 中添加:
if trajectory_mode == "bezier":
    # 用户点击控制点
    # 生成贝塞尔曲线上的 49 个等距点
    points = bezier_interpolate(control_points, num_frames=49)
```

---

## 3. 新分辨率支持

### 影响范围

| 组件 | 需要修改的地方 | 说明 |
|------|---------------|------|
| 背景预处理 | `get_bgvideo.py` — `720×480` 硬编码 | 改为参数 |
| 前景画布 | `get_fgmask.py`, `video_utils.py` — `576×576` | 改为参数 |
| VAE latent | 自动适应（H/8, W/8） | 无需修改 |
| ERoPE | `prepare_rotary_positional_embeddings()` | 自动根据 grid_size 计算 |
| Pipeline | `sample_height`, `sample_width` 参数 | 传入新分辨率 |
| 训练 | `train.sh` 中 `--height`, `--width` | 修改参数 |

### 注意事项

- CogVideoX-5B 预训练于 480×720，大幅偏离可能导致质量下降
- 分辨率必须能被 VAE 下采样因子（8）整除
- 更高分辨率需要更多 GPU 显存（VRAM >= 40GB for 480×720）

---

## 4. 接入其他 Backbone

### 当前架构依赖

```
CogVideoX-5B-I2V
├── VAE: AutoencoderKLCogVideoX (16 通道 latent)
├── Transformer: CogVideoXTransformer3D (42 层 DiT)
├── Text Encoder: T5-XXL
└── Scheduler: DPM/DDIM
```

### 替换 Backbone 的最小改动路径

#### 方案 A: 替换为其他 CogVideo 版本
- **改动量**: 小
- **步骤**:
  1. 更新 `pretrained_model_name_or_path` 指向新模型
  2. 调整 `in_channels` 如果 latent 维度不同
  3. 重新训练 Branch 和 Transformer

#### 方案 B: 替换为完全不同的视频扩散模型（如 SVD, AnimateDiff）
- **改动量**: 大
- **需要修改的核心文件**:

| 文件 | 修改内容 |
|------|---------|
| `branch_cogvideox.py` | 重写 Branch 结构匹配新 backbone 的层数和维度 |
| `cogvideox_transformer_3d_fg3b_sep.py` | 替换为新 backbone 的 Transformer + 分离 embedding |
| `embeddings.py` | 适配新模型的位置编码方案 |
| `pipeline_*.py` | 重写推理管线 |
| 训练脚本 | 适配新模型的数据流和 loss |

#### 方案 C: 添加 LoRA 适配
- **改动量**: 中
- **优势**: 不需要全参数训练，减少显存
- **入口**: 在 `CogVideoXTransformer3D3BModel_sep` 的 attention 层添加 LoRA

---

## 5. 新数据集接入

### 当前数据格式

```
VideoComp/
├── GTs/              # GT 视频 (合成结果)
├── fg/               # 前景元素 (576×576)
├── filtered_mask/    # 二值 mask
├── filtered_masked_video/ # 背景 × (1-mask)
├── inpainted_sum/    # 去前景背景
└── white_video.mp4   # swap 模式用
```

**CSV 格式**: `path, start_frame, end_frame, fps, caption`

### 接入新数据集

1. **准备数据**: 将新数据集转换为上述目录结构
2. **创建 CSV**: 生成 `train.csv` 和 `val.csv`
3. **修改路径**: `train.sh` 中 `--instance_data_root` 和 `--meta_file_path`
4. **检查**: 确保所有视频分辨率/帧数/FPS 一致

### 数据集制作工具链

```
原始视频对 (前景+背景)
    ↓
[get_bgvideo.py] 背景标准化
    ↓
[get_fgmask.py]  前景分割 → mask + element
    ↓
生成 masked_video = bg × (1-mask)
    ↓
生成 inpainted_bg (可用任意 inpainting 模型)
    ↓
组织为 VideoComp 目录结构 + CSV
```

---

## 6. 功能扩展建议

### 6.1 多目标合成
- **当前限制**: 单前景目标
- **扩展方向**: 多次推理叠加，或修改 pipeline 支持多 mask 通道
- **入口**: `testinput.py::generate_video()` — 循环处理多个前景

### 6.2 文本引导的合成控制
- **当前**: 文本 prompt 主要用于风格描述
- **扩展方向**: 支持 "put the dog on the left side" 等空间指令
- **入口**: prompt encoding 层面 + 交叉注意力修改

### 6.3 实时预览
- **当前**: 全量推理后才能看结果
- **扩展方向**: 减少 inference steps + 低分辨率预览
- **入口**: `testinput.py` — 添加 `--preview_mode` 参数（如 5 步、256×384）

### 6.4 视频长度扩展
- **当前限制**: 49 帧 (~6 秒 @8fps)
- **扩展方向**: 滑动窗口 + 帧间一致性约束
- **入口**: Pipeline 层面添加 chunk 处理逻辑
