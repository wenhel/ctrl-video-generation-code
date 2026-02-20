# GenCompositor 论文→代码对照

> 论文: "GenCompositor: Generative Video Compositing with Diffusion Transformer" (ICLR 2026)

---

## 方法概述

论文提出基于 DiT（Diffusion Transformer）的生成式视频合成框架，包含四个核心创新点：

1. **背景保留分支（Background Preservation Branch）** — masked token injection
2. **DiT 融合块（DiT Fusion Block）** — full self-attention
3. **扩展旋转位置编码（ERoPE）** — Extended Rotary Position Embedding
4. **前景增强（Foreground Augmentation）** — 训练时数据增强

---

## 1. 背景保留分支 (Background Preservation Branch)

**论文描述**: 轻量级 DiT 分支，通过 masked token injection 保持背景一致性。分支从被 mask 的背景视频中提取上下文特征，注入到主 Transformer 的每一层。

### 代码落点

| 组件 | 文件 | 位置 |
|------|------|------|
| **Branch 模型定义** | `diffusers/src/diffusers/models/branch_cogvideox.py` | `CogvideoXBranchModel` 类 |
| **Branch 配置** | 同上 | 30 层 Transformer，in_channels=16，zero-init |
| **Branch 初始化** | 同上 | `from_transformer()` — 从预训练 Transformer 初始化 |
| **Zero Module** | 同上 | `zero_module()` — 分支输出层初始化为零，确保训练稳定 |
| **Branch 推理调用 (推理)** | `infer/testinput.py` | `generate_video()` 中加载 branch 并传入 pipeline |
| **Branch 推理调用 (训练)** | `train/train_cogvideox_compositing_sep.py` | 训练循环中 `branch(...)` 调用 |
| **Masked token injection** | `diffusers/.../pipeline_cogvideox_inpainting_i2v_branch3_sep.py` | `hidden_states = where(mask==False, hidden + branch_samples[i], hidden)` |

### 关键实现细节

```python
# Branch 输入: masked_video_latent + mask (17 通道)
latent_branch_input = torch.cat([masked_video_latents, mask[:, :, :1, :, :]], dim=-3)

# Branch 输出: 30 层 block_samples
branch_block_samples = self.branch(
    image_embeds=latent_branch_input,
    encoder_hidden_states=prompt_embeds,
    timestep=timestep,
    conditioning_scale=branch_scale
)

# 注入: 仅在背景区域（mask==False）添加 branch 特征
hidden_states = torch.where(
    masks == False,
    hidden_states + branch_block_samples[i],
    hidden_states
)
```

---

## 2. DiT 融合块 (DiT Fusion Block)

**论文描述**: 使用 full self-attention 将前景和背景 token 统一处理，实现自然融合。

### 代码落点

| 组件 | 文件 | 位置 |
|------|------|------|
| **分离 Transformer** | `diffusers/.../cogvideox_transformer_3d_fg3b_sep.py` | `CogVideoXTransformer3D3BModel_sep` |
| **分离 Patch Embedding** | `diffusers/.../embeddings.py` | `CogVideoXPatchEmbed_sep` 类 |
| **注意力处理器** | `diffusers/.../attention_processor.py` | `CogVideoXAttnProcessor2_0` |
| **Block 结构** | `diffusers/.../cogvideox_transformer_3d_fg3b.py` | `CogVideoXBlock` |

### 关键实现细节

```python
# CogVideoXPatchEmbed_sep: 双投影头
# 前 32 通道 (noisy_latent + cond_latent) → proj()
# 后 16 通道 (fg_latent) → fg_proj()
image_embeds = self.proj(image_embeds[:, :, :32, ...])     # 背景路径
fg_embeds = self.fg_proj(image_embeds[:, :, -16:, ...])    # 前景路径

# 拼接所有 token 进行 full self-attention
embeds = torch.cat([text_embeds, image_embeds, fg_embeds], dim=1)

# Attention: 所有 token 统一参与 self-attention
# text + background + foreground → Q, K, V → attention → 输出
```

---

## 3. ERoPE (Extended Rotary Position Embedding)

**论文描述**: 扩展的 3D 旋转位置编码，支持不同空间布局的前景和背景视频融合。

### 代码落点

| 组件 | 文件 | 位置 |
|------|------|------|
| **3D RoPE 生成** | `diffusers/.../embeddings.py` | `get_3d_rotary_pos_embed()` |
| **2D/1D RoPE 基础** | 同上 | `get_2d_rotary_pos_embed()`, `get_1d_rotary_pos_embed()` |
| **RoPE 应用** | `diffusers/.../attention_processor.py` | `apply_rotary_emb()` |
| **位置编码准备 (训练)** | `train/train_cogvideox_compositing_sep.py` | `prepare_rotary_positional_embeddings()` |
| **位置编码准备 (推理)** | Pipeline 内部 | 同名函数 |

### 关键实现细节

```python
# 3D RoPE: 时间 + 高度 + 宽度 三维分解
def get_3d_rotary_pos_embed(embed_dim, crops_coords, grid_size, temporal_size, ...):
    # 维度分配: dim_t = dim//4, dim_h = dim_w = dim//8 * 3
    # 时间维度: 1D RoPE
    # 空间维度: 2D RoPE (H × W)
    # 返回: (cos, sin) 对

# 训练时: 分别为主输入和拼接输入准备不同的位置编码
image_rotary_emb = prepare_rotary_positional_embeddings(...)        # 主序列
image_rotary_emb_for_concat = prepare_rotary_positional_embeddings(...)  # 含 branch 维度

# Attention 中应用:
query[:, text_seq_length:] = apply_rotary_emb(query[:, text_seq_length:], emb)
key[:, text_seq_length:] = apply_rotary_emb(key[:, text_seq_length:], emb)
```

---

## 4. 前景增强 (Foreground Augmentation)

**论文描述**: 训练时对前景 mask 进行数据增强，提升模型对不同 mask 形状的鲁棒性。

### 代码落点

| 组件 | 文件 | 位置 |
|------|------|------|
| **Mask 变换主函数** | `train/mask_process.py` | `transform_video_masks()` |
| **随机笔刷生成** | 同上 | `generate_random_brush()` |
| **变换调用** | `train/train_cogvideox_compositing_sep.py` | `MyWebDataset.__call__()` 中调用 |
| **Swap 增强** | 同上 | `VideoInpaintingDataset.__getitem__()` — 20% 概率交换前景为白视频 |

### 五种 Mask 变换模式

| 模式 | 概率 | 方法 |
|------|------|------|
| `brush` | 0.25 | 形态学腐蚀/膨胀 + 可选高斯模糊 |
| `random_brush` | 0.25 | 随机笔刷描边（多边形 + 椭圆） |
| `rect` | 0.25 | 从 bbox 提取矩形 |
| `ellipse` | 0.125 | 从 bbox 提取椭圆 |
| `circle` | 0.125 | 从 bbox 提取圆形 |

```python
# Collate 中调用 (30% 概率)
if random.random() < mask_transform_prob:
    masks = transform_video_masks(
        masks,
        p_brush=0.25, p_rect=0.25,
        p_ellipse=0.125, p_circle=0.125,
        p_random_brush=0.25
    )
```

---

## 5. 其他论文→代码对应

### VideoComp 数据集

| 论文描述 | 代码位置 |
|---------|---------|
| 61K 视频对 | `train/train_demo.csv` (格式示例) |
| GT / FG / Mask / Masked Video / Inpainted BG | `VideoComp/` 目录结构 |
| 数据加载 | `VideoInpaintingDataset` 类 |

### CogVideoX-5B-I2V 基础模型

| 论文描述 | 代码位置 |
|---------|---------|
| 基于 CogVideoX DiT | `diffusers/.../cogvideox_transformer_3d.py` (原版) |
| I2V inpainting | 多个 pipeline 文件 |
| VAE 编解码 | `diffusers/.../autoencoder_kl_cogvideox.py` |
| DPM Scheduler | `diffusers/.../scheduling_dpm_cogvideox.py` |

### 推理流程

| 论文描述 | 代码位置 |
|---------|---------|
| 背景/前景视频输入 | `infer/get_bgvideo.py`, `infer/get_fgmask.py` |
| SAM2 前景分割 | `infer/get_fgmask.py` — Grounding DINO + SAM2 |
| 用户轨迹控制 | `infer/get_movemask.py`, `infer/usr.py` |
| 合成输出 | `infer/testinput.py` → `CogVideoXI2VTriInpaintPipeline_sep` |

### 训练策略

| 论文描述 | 代码位置 | 值 |
|---------|---------|-----|
| Inpainting loss 加权 | `main()` 训练循环 | 5.0 |
| 梯度裁剪 | 同上 | max_norm=1.0 |
| 混合精度 | `accelerate_config` | bf16 |
| 分布式训练 | 同上 | DeepSpeed ZeRO-2, 8 GPU |

---

## 架构对照图

```
┌─────────────────────────────────────────────────────┐
│                   论文 Figure 2                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Background Video ─→ [VAE] ─→ Masked Latent         │
│       (mask)              ↓                          │
│                    ┌─────────────┐                   │
│                    │   Branch    │ ← branch_cogvideox │
│                    │  (30层DiT)  │     .py            │
│                    └──────┬──────┘                   │
│                           │ block_samples            │
│                           ↓ (mask==False区域注入)      │
│  Noisy Latent ──→ ┌──────────────┐                   │
│  + Cond Latent     │  Main DiT    │ ← transformer_   │
│  + FG Latent  ──→ │  (42层)      │    3d_fg3b_sep.py │
│     ↑              │  ERoPE      │ ← embeddings.py   │
│   fg_proj          │  Fusion Attn│ ← attention_      │
│   (分离嵌入)        └──────┬──────┘    processor.py   │
│                           │                          │
│                    [VAE Decode]                       │
│                           ↓                          │
│                   Composed Video                     │
└─────────────────────────────────────────────────────┘
```
