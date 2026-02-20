# GenCompositor 数据流文档

---

## 一、推理数据流

### 总体流程

```
背景视频 + 前景视频 + 用户轨迹
    ↓
预处理（分辨率/帧数标准化）
    ↓
前景分割（SAM2 + Grounding DINO）
    ↓
轨迹绘制 + Mask 视频生成
    ↓
CogVideoX Inpainting 合成
    ↓
最终合成视频
```

### Step 1: 背景视频预处理 — `get_bgvideo.py`

```
输入: bg_video.mp4 (任意分辨率/帧数)
    ↓ resize_and_save_frames()
    ├─ 读取所有帧
    ├─ 缩放至 720×480
    ├─ 帧跳跃（200-300帧取2, 300-400帧取3, 400+帧取4）
    ├─ 不足49帧 → ping-pong 补帧（前进→后退→前进…）
    └─ 输出: bg_video.mp4 [49帧, 720×480, 12fps]
```

### Step 2: 前景分割 — `get_fgmask.py`

```
输入: fg_video.mp4 + text_prompt (目标物体描述)
    ↓
    ├─ 1. ffmpeg 提取 49 帧 → frames/ 目录
    ├─ 2. Grounding DINO 检测第一帧中的目标 → bbox
    ├─ 3. SAM2 Image Predictor: bbox → 第一帧 mask
    ├─ 4. SAM2 Video Predictor: mask 传播至全部 49 帧
    ├─ 5. 保存 per-frame mask → mask/ 目录
    └─ 6. create_fg_video():
         ├─ 二值化 + 形态学开运算 (kernel 9×9)
         ├─ 检查有效性（面积、边缘触碰）
         ├─ 自适应缩放（2x/4x 上采样 或 下采样）
         ├─ 居中放置于 576×576 白色画布
         └─ 输出:
              ├─ fg_mask.mp4 [576×576, mask视频]
              └─ fg_element.mp4 [576×576, 前景元素视频]
```

### Step 3: 轨迹与 Mask 视频 — `get_movemask.py` / `usr.py`

```
输入: bg_video.mp4 (背景) + fg_element.mp4 (前景) + 用户交互
    ↓
    ├─ 1. draw_and_save_trajectory():
    │     ├─ 显示背景第一帧
    │     ├─ 用户鼠标绘制轨迹
    │     ├─ 单点 → LK 光流追踪
    │     ├─ 少于帧数 → 线性插值
    │     ├─ 多于帧数 → 均匀采样
    │     └─ 输出: trajectory.txt [(x,y) × 49帧] + trajectory.png
    │
    └─ 2. generate_mask_video_with_trajectory():
          ├─ 读取 fg_element 提取轮廓
          ├─ 计算 bbox，应用 scale 参数
          ├─ 按轨迹坐标逐帧放置前景
          ├─ 支持 center/bottom 对齐
          └─ 输出: mask_video.mp4 [720×480, 灰度 mask]
```

### Step 4: 视频合成 — `testinput.py::generate_video()`

```
输入:
  ├─ bg_video.mp4 [49帧, 720×480]
  ├─ mask_video.mp4 [49帧, 720×480, 灰度]
  └─ fg_element.mp4 [49帧, 576×576]

    ↓ read_video_with_mask()
    ├─ 1. 加载视频 → tensor
    ├─ 2. Mask 处理:
    │     ├─ 腐蚀 (3×3 kernel)
    │     ├─ 连通域过滤 (面积 < 0.01% 去除)
    │     ├─ 膨胀
    │     └─ 高斯模糊 (51×51, σ=10)
    ├─ 3. 前景缩放至背景尺寸，居中 padding
    └─ 输出:
         ├─ video [T,H,W,3] 背景视频
         ├─ masked_video [T,H,W,3] 背景×(1-mask)
         ├─ binary_masks [T,H,W,1]
         ├─ fg_resized [T,H,W,3] 前景 RGB
         └─ fgy_resized [T,H,W,1] 前景灰度

    ↓ Pipeline 推理
    ├─ 1. 加载模型:
    │     ├─ CogVideoXI2VTriInpaintPipeline_sep (主 pipeline)
    │     ├─ CogvideoXBranchModel (背景保留分支)
    │     └─ CogVideoXTransformer3D3BModel_sep (主 Transformer)
    │
    ├─ 2. 数据预处理:
    │     ├─ video → [B,T,C,H,W] 归一化至 [-1,1]
    │     ├─ 第一帧提取 → image_latent (VAE 编码)
    │     └─ 帧下采样 (down_sample_fps)
    │
    ├─ 3. Pipeline.__call__():
    │     ├─ VAE 编码: video → latent [B,F,C,H/8,W/8]
    │     ├─ VAE 编码: masked_video → cond_latent
    │     ├─ VAE 编码: fg_video → fg_latent
    │     ├─ Mask 插值至 latent 空间
    │     ├─ 拼接: [cond_latent | mask] → branch 输入
    │     │
    │     ├─ Branch 推理:
    │     │   └─ branch(masked_latent) → block_samples[0..29]
    │     │
    │     ├─ 加噪: noisy_latent = latent + noise × σ
    │     │
    │     ├─ Transformer 推理:
    │     │   ├─ 输入: [noisy_latent | cond_latent | fg_latent] (通道拼接)
    │     │   ├─ ERoPE 位置编码
    │     │   ├─ Block 内: hidden += branch_samples[i] (仅背景区域)
    │     │   └─ 输出: predicted noise/velocity
    │     │
    │     ├─ Scheduler step (DPM/DDIM)
    │     └─ 循环 num_inference_steps 次
    │
    ├─ 4. VAE 解码: latent → video [B,T,3,H,W]
    └─ 5. 保存输出视频

输出: composed_video.mp4 [49帧, 720×480]
```

---

## 二、训练数据流

### 总体流程

```
VideoComp 数据集 (CSV + 视频文件)
    ↓
VideoInpaintingDataset (数据加载)
    ↓
MyWebDataset (Collate + 增强)
    ↓
VAE 编码 + T5 文本编码
    ↓
扩散前向 (加噪 + Branch + Transformer)
    ↓
Loss 计算 (重建 + Inpainting)
    ↓
反向传播 + 优化器更新
```

### 数据集结构

```
VideoComp/
├── GTs/           # Ground Truth 视频 (合成后完整视频)
├── fg/            # 前景元素视频 (576×576)
├── filtered_mask/ # 二值 mask 视频
├── filtered_masked_video/ # 背景 × (1-mask) 视频
├── inpainted_sum/ # 去前景后的纯背景视频
└── white_video.mp4  # 用于 swap 模式的全白视频
```

### 数据加载 — `VideoInpaintingDataset.__getitem__()`

```
CSV 行: path=40029.mp4, start=0, end=100, fps=16, caption="..."
    ↓
    ├─ ffmpeg 解码 GT 视频 → [T,H,W,3] uint8
    ├─ ffmpeg 解码 fg 视频 → [T,H,W,3] uint8
    ├─ ffmpeg 解码 masked_video → [T,H,W,3] uint8
    ├─ ffmpeg 解码 mask → [T,H,W,1] uint8
    │
    ├─ 20% 概率: swap 模式
    │   └─ fg 替换为 white_video, mask 反转
    │
    └─ 输出 dict:
         ├─ instance_video [T,H,W,3]     # GT 视频
         ├─ fg_video [T,H,W,3]           # 前景
         ├─ masked_video [T,H,W,3]       # 背景 masked
         ├─ masks [T,H,W,1]              # 二值 mask
         └─ instance_prompt: str          # 文字描述
```

### Collate — `MyWebDataset.__call__()`

```
batch of dicts
    ↓
    ├─ 1. Prompt 分词:
    │     ├─ 10% 概率置空 (empty prompt dropout)
    │     └─ T5 Tokenizer → input_ids [B, seq_len]
    │
    ├─ 2. 视频处理:
    │     ├─ 找最近标准分辨率
    │     ├─ 缩放 + 中心裁剪至 480×720
    │     └─ 归一化: [0,255] → [-1,1]
    │
    ├─ 3. Mask 变换 (30% 概率):
    │     └─ transform_video_masks():
    │           ├─ brush: 形态学腐蚀/膨胀 + 高斯模糊
    │           ├─ random_brush: 随机笔刷描边
    │           ├─ rect: 从 bbox 提取矩形
    │           ├─ ellipse: 从 bbox 提取椭圆
    │           └─ circle: 从 bbox 提取圆形
    │
    └─ 输出 batch:
         ├─ pixel_values [B,T,C,H,W]              # GT 视频 float32
         ├─ conditioning_pixel_values [B,T,C,H,W]  # masked 背景
         ├─ masks [B,T,1,H,W]                      # 二值 mask
         ├─ fg_pixel_values [B,T,C,H,W]            # 前景视频
         └─ input_ids [B, seq_len]                  # 文本 token
```

### 训练步骤 — `main()` 训练循环

```
batch
    ↓
    ├─ 1. VAE 编码 (全部冻结, 无梯度):
    │     ├─ pixel_values → latent [B,F,16,H/8,W/8]     # F = ceil(T/4)
    │     ├─ conditioning → cond_latent [B,F,16,H/8,W/8]
    │     ├─ fg_pixel_values → fg_latent [B,F,16,H/8,W/8]
    │     ├─ 第一帧(加噪) → image_latent [B,1,16,H/8,W/8]
    │     └─ image_latent 拼接至 cond_latent 第一帧
    │
    ├─ 2. Mask 处理:
    │     ├─ 插值至 latent 分辨率 [B,F,1,H/8,W/8]
    │     ├─ 二值化 (> 0.5)
    │     └─ 拼接: [cond_latent | mask_latent] → branch_input
    │
    ├─ 3. 文本编码:
    │     └─ T5 Encoder(input_ids) → prompt_embeds [B, seq_len, 4096]
    │
    ├─ 4. 扩散前向:
    │     ├─ 采样随机时间步 t
    │     ├─ 采样高斯噪声 ε
    │     └─ noisy_latent = latent + ε × σ(t)
    │
    ├─ 5. 位置编码:
    │     └─ prepare_rotary_positional_embeddings()
    │           → image_rotary_emb (cos, sin)
    │           → image_rotary_emb_for_concat (含 branch 维度)
    │
    ├─ 6. Branch 推理 (冻结):
    │     ├─ 输入: [cond_latent | mask]
    │     └─ 输出: block_samples[0..29] — 每层的控制特征
    │
    ├─ 7. Transformer 前向 (可训练):
    │     ├─ 输入: [noisy_latent | cond_latent | fg_latent] → 48 通道
    │     ├─ CogVideoXPatchEmbed_sep:
    │     │     ├─ 前 32 通道 → proj() → image_embeds
    │     │     └─ 后 16 通道 → fg_proj() → fg_embeds
    │     ├─ 拼接: [text_embeds, image_embeds, fg_embeds]
    │     ├─ 每个 Block:
    │     │     ├─ Self-Attention (text + image + fg, 含 ERoPE)
    │     │     ├─ FFN
    │     │     └─ += branch_samples[i] (仅 mask==False 区域)
    │     └─ 输出: predicted velocity
    │
    ├─ 8. Loss 计算:
    │     ├─ target = scheduler.get_velocity(latent, noise, t)
    │     ├─ recon_loss = MSE(pred, target)  # 全序列
    │     ├─ inpaint_loss = MSE(pred × mask, target × mask)  # 仅 mask 区域
    │     └─ total_loss = recon_loss + 5.0 × inpaint_loss
    │
    └─ 9. 反向传播:
          ├─ accelerator.backward(total_loss)
          ├─ gradient clipping (max_norm=1.0)
          ├─ optimizer.step() (AdamW, lr=2e-5)
          └─ scheduler.step() (cosine annealing)
```

### Checkpoint & 验证

```
每 1024 步:
    ├─ 保存 Transformer 权重
    └─ 保留最近 3 个 checkpoint

每 256 步:
    ├─ 加载推理 pipeline
    ├─ 在验证集生成 2 个样本
    └─ 上传 wandb (视频 + loss 曲线)
```

---

## 三、关键张量形状一览

| 张量 | 形状 | 说明 |
|------|------|------|
| 输入视频 | `[B, T, 3, 480, 720]` | T=49 帧 |
| VAE latent | `[B, F, 16, 60, 90]` | F=ceil(49/4)=13 |
| Mask (像素) | `[B, T, 1, 480, 720]` | 二值 |
| Mask (latent) | `[B, F, 1, 60, 90]` | 插值后二值化 |
| Branch 输入 | `[B, F, 17, 60, 90]` | 16ch latent + 1ch mask |
| Transformer 输入 | `[B, F, 48, 60, 90]` | 16 noisy + 16 cond + 16 fg |
| Prompt embeds | `[B, seq_len, 4096]` | T5 编码输出 |
| Image rotary emb | `[seq_len, dim/2]` × 2 | cos/sin 对 |
| Block samples | `[B, T', C, H', W']` × 30 | 每层 branch 输出 |
