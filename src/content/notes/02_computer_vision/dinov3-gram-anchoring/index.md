---
title: "What's new in DINOv3: scaling self-supervised vision and fixing dense features with Gram anchoring"
slug: dinov3-gram-anchoring
date: 2026-05-06
topic: Computer Vision
tags: [dinov3, dinov2, self-supervised-learning, gram-matrix, rope, vision-transformer, dense-features]
summary: "Why DINOv3's patch features got worse with longer training, how the Gram anchoring loss repairs them, and what else changed from DINOv2."
sources:
  - title: "DINOv3"
    author: "Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab et al. (Meta AI)"
    url: "https://arxiv.org/abs/2508.10104"
    type: paper
  - title: "DINOv2: Learning Robust Visual Features without Supervision"
    author: "Maxime Oquab et al."
    url: "https://arxiv.org/abs/2304.07193"
    type: paper
  - title: "Vision Transformers Need Registers"
    author: "Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski"
    url: "https://arxiv.org/abs/2309.16588"
    type: paper
  - title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    author: "Jianlin Su et al."
    url: "https://arxiv.org/abs/2104.09864"
    type: paper
  - title: "A Neural Algorithm of Artistic Style"
    author: "Leon A. Gatys, Alexander S. Ecker, Matthias Bethge"
    url: "https://arxiv.org/abs/1508.06576"
    type: paper
  - title: "facebookresearch/dinov3"
    author: "Meta AI"
    url: "https://github.com/facebookresearch/dinov3"
    type: repo
  - title: "DINOv3 model documentation"
    author: "Hugging Face Transformers"
    url: "https://huggingface.co/docs/transformers/model_doc/dinov3"
    type: docs


status: draft
---

DINOv3 (Meta AI, August 2025) asks a simple question: what happens if we take the DINOv2 recipe and make everything much bigger? The answer was surprising. Image-level quality kept improving, but the patch-level features, the ones you need for segmentation and depth, slowly got worse. DINOv3 is mainly the story of finding that problem and fixing it with a new loss called **Gram anchoring**. This note assumes you know the DINOv2 basics (student, EMA teacher, DINO + iBOT + KoLeo losses); see my DINOv2 note for those.

## What stays the same

The core is still DINOv2: a student ViT, an EMA teacher, multi-crop augmentation (2 global and 8 local crops), and three losses. The first training phase optimizes

$$
\mathcal{L}_{\text{Pre}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + 0.1\,\mathcal{L}_{\text{DKoLeo}}
$$

where $\mathcal{L}_{\text{DINO}}$ matches image-level outputs, $\mathcal{L}_{\text{iBOT}}$ matches masked patch outputs and $\mathcal{L}_{\text{DKoLeo}}$ is a KoLeo regularizer computed across GPUs (the "D" stands for distributed). As in DINOv2, teacher outputs are normalized with Sinkhorn-Knopp instead of DINO's centering.

## What changes before the main fix

**Scale.** The main model is a ViT-7B (about 6.7 billion parameters, patch size 16, 4096-dimensional embeddings). It is trained on **LVD-1689M**, about 1.7 billion images selected from roughly 17 billion public Instagram images. Selection combines hierarchical k-means clustering (so rare visual concepts are not drowned out) with retrieval of images similar to curated datasets.

**Constant schedules.** DINOv2 used cosine schedules, which require you to fix the training length in advance. DINOv3 drops them. After a short warmup, the learning rate, weight decay and teacher momentum stay constant. You can keep training as long as downstream results improve.

**2D RoPE with box jittering.** Learned positional embeddings are replaced by rotary positional embeddings (RoPE). Instead of adding a position vector to each token, RoPE rotates the query and key vectors by angles that depend on each patch's $(x, y)$ position. The attention score between two patches then depends on their *relative* position, which helps when the image size at test time differs from training. Patch coordinates are placed in a normalized box $[-1, 1]$. During training the box is randomly rescaled to $[-s, s]$ with $s \in [0.5, 2]$, so the model does not overfit to one grid size or aspect ratio.

```python
import torch

def patch_coords(h, w, train=True):
    ys = torch.linspace(-1, 1, h)                 # normalized patch centers
    xs = torch.linspace(-1, 1, w)
    coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (h, w, 2)
    if train:                                     # RoPE box jittering
        s = torch.empty(1).uniform_(0.5, 2.0)
        coords = coords * s
    return coords                                 # fed to the RoPE sin/cos tables
```

**Register tokens.** Four extra learnable tokens sit next to `[CLS]`. They give the network a place to store global information, which removes the high-norm "artifact" patches that otherwise appear in attention maps (Darcet et al., 2024).

## The problem: dense features degrade with long training

The authors tracked two numbers during training:

- **ImageNet linear classification** using the `[CLS]` token (a global task);
- **Pascal VOC linear segmentation** using the patch tokens (a dense task).

Classification kept improving. Segmentation peaked at around 200k iterations and then declined; for the ViT-7B it fell below its early level. Looking at cosine-similarity maps between one patch and all others showed why. Early in training the maps were sharp and local: a patch on a cat's ear was similar to other cat patches only. Later, many unrelated patches also became similar to the reference. The patch features were losing their locality and drifting towards a global summary of the image. The image-level objective and the patch-level quality were pulling in different directions.

## The fix: Gram anchoring

Let $X_S \in \mathbb{R}^{P \times d}$ be the L2-normalized patch features of the student for one image ($P$ patches, $d$ dimensions). The **Gram matrix** $X_S X_S^\top \in \mathbb{R}^{P \times P}$ holds the cosine similarity of every pair of patches. It describes the *structure* of the feature map: which patches belong together and which do not.

DINOv3 keeps an extra network, the **Gram teacher**, taken from an earlier checkpoint of the EMA teacher when the patch features were still clean. With $X_G$ its normalized patch features on the same image, the new loss is

$$
\mathcal{L}_{\text{Gram}} = \left\lVert X_S X_S^\top - X_G X_G^\top \right\rVert_F^2
$$

where $\lVert \cdot \rVert_F$ is the Frobenius norm (the square root of the sum of squared entries).

The key design choice is that the loss compares *similarities*, not features. If we forced $X_S = X_G$, the student could never get better than the old checkpoint. Matching only the Gram matrices lets the features rotate or change in any way that keeps the pairwise relationships. In plain words: "keep the same patch structure, but feel free to encode it better."

```python
import torch.nn.functional as F

def gram(x):                          # x: (B, P, d) patch features
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)      # (B, P, P) cosine similarities

def gram_loss(student_patches, gram_teacher_patches):
    with torch.no_grad():
        target = gram(gram_teacher_patches)
    return ((gram(student_patches) - target) ** 2).sum(dim=(1, 2)).mean()
```

How it is used in practice:

- The loss is added in a **refinement phase** that starts after the main pretraining (1M iterations), and it is computed on global crops only.
- The Gram teacher is refreshed every 10k iterations by copying the current EMA teacher.
- Applied late, it still **repairs** features that had already degraded.
- A further gain comes from feeding the Gram teacher images at twice the resolution and downsampling its feature map. Its patch features are then smoother and more detailed.

The refinement objective is

$$
\mathcal{L}_{\text{Ref}} = w_D\,\mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + w_{DK}\,\mathcal{L}_{\text{DKoLeo}} + w_{\text{Gram}}\,\mathcal{L}_{\text{Gram}}
$$

with weights $w$ set by the authors. They observed that the Gram loss makes the iBOT loss drop faster but barely affects the DINO loss. This suggests the Gram and iBOT losses shape features in a similar (local) way, while the DINO loss acts differently.

The idea of a Gram matrix will look familiar from neural style transfer (Gatys et al., 2015), where it captures texture by correlating *channels*. In DINOv3 the Gram matrix correlates *patches*, so it captures spatial structure instead.

## After the main training

1. **High-resolution adaptation.** A short phase with mixed, larger crop sizes (global crops of 512 and 768 pixels), with Gram anchoring still on. This lets the model handle high-resolution inputs well.
2. **Distillation.** The 7B model is used as a frozen teacher to train smaller models: ViT-S, S+, B, L, H+ and ConvNeXt variants. Most people will use these.
3. **Extras.** A text encoder aligned to the features (dino.txt) and a separate model trained on satellite imagery.

## Using a released model

The small model `facebook/dinov3-vits16-pretrain-lvd1689m` has 12 blocks, 6 heads, 384-dimensional tokens and a standard GELU MLP (1536 hidden units). The SwiGLU feed-forward layers are only in the large 7B teacher. For a 224×224 input with patch size 16 there are $14 \times 14 = 196$ patches. The output sequence is 1 `[CLS]` + 4 registers + 196 patches = 201 tokens, so you must skip the registers when you pick out patch tokens:

```python
from transformers import AutoImageProcessor, AutoModel
import torch

name = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name)

inputs = processor(images=image, return_tensors="pt")      # (1, 3, 224, 224)
with torch.inference_mode():
    out = model(**inputs)

h = out.last_hidden_state                                   # (1, 201, 384)
n_reg = model.config.num_register_tokens                    # 4
cls = h[:, 0]                                               # global descriptor
patches = h[:, 1 + n_reg:].unflatten(1, (14, 14))           # (1, 14, 14, 384)
```

## Key takeaways

- DINOv3 scales the DINOv2 recipe to a 7B ViT and 1.7B curated images, with constant learning-rate schedules.
- At this scale, long training improves global features but damages patch features.
- Gram anchoring fixes this by matching the student's patch-to-patch similarity matrix to that of an earlier checkpoint, without freezing the features themselves.
- 2D RoPE with box jittering and register tokens make the features more robust to resolution and cleaner.
- In practice you use a distilled model and skip the 4 register tokens when extracting patch features.

## Resources

- Siméoni et al., "DINOv3", 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Darcet et al., "Vision Transformers Need Registers", ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Gatys, Ecker, Bethge, "A Neural Algorithm of Artistic Style", 2015. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
- Official code and weights: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Hugging Face docs: [DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3)
