---
title: "What's new in DINOv3: scaling self-supervised vision and fixing dense features with Gram anchoring"
slug: dinov3-gram-anchoring
date: 2026-05-06
topic: Computer Vision
tags: [dinov3, dinov2, self-supervised-learning, gram-matrix, rope, vision-transformer, dense-features]
summary: "Why DINOv3's patch features got worse the longer it trained, how the Gram anchoring loss fixes that, and what else changed since DINOv2."
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

DINOv3 (Meta AI, August 2025) starts from a simple question: what if we take the DINOv2 recipe and just make everything much bigger? The answer surprised me. The image-level features kept getting better, but the patch-level features (the ones you need for segmentation and depth) slowly got worse. So DINOv3 is mostly the story of spotting that problem and fixing it with a new loss called **Gram anchoring**. I'm assuming the DINOv2 basics here (student, EMA teacher, DINO + iBOT + KoLeo losses). My [DINOv2 note](/notes/dinov2-self-supervised-features/) covers those.

## What stays the same

The core is still DINOv2: a student ViT, a teacher that is an EMA (a slow moving average) of the student, multi-crop augmentation (2 global and 8 local crops), and three losses. The first training phase optimizes

$$
\mathcal{L}_{\text{Pre}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + 0.1\,\mathcal{L}_{\text{DKoLeo}}
$$

Here $\mathcal{L}_{\text{DINO}}$ matches the image-level outputs, $\mathcal{L}_{\text{iBOT}}$ matches the outputs for masked patches, and $\mathcal{L}_{\text{DKoLeo}}$ is a KoLeo regularizer computed across GPUs (the "D" means distributed). Like DINOv2, the teacher outputs are normalized with Sinkhorn-Knopp instead of the centering trick from the original DINO.

## What changes before the main fix

**Scale.** The main model is a ViT-7B: about 6.7 billion parameters, patch size 16, 4096-dimensional embeddings. It's trained on **LVD-1689M**, about 1.7 billion images picked from roughly 17 billion public Instagram images. The picking uses two things: hierarchical k-means clustering, so rare visual concepts don't get drowned out, and retrieval of images that look like those in curated datasets.

**Constant schedules.** DINOv2 used cosine schedules, which means you have to decide the training length up front. DINOv3 drops them. After a short warmup, the learning rate, weight decay and teacher momentum just stay constant. So you can keep training for as long as the downstream results keep improving.

**2D RoPE with box jittering.** The learned positional embeddings are replaced by rotary positional embeddings (RoPE). Instead of adding a position vector to each token, RoPE rotates the query and key vectors by angles that depend on each patch's $(x, y)$ position. The way I think about it: after the rotation, the attention score between two patches only depends on where they are *relative* to each other. That helps when the test image size is different from the training size. Patch coordinates live in a normalized box $[-1, 1]$. During training the box is randomly stretched to $[-s, s]$ with $s \in [0.5, 2]$, so the model doesn't get stuck on one grid size or aspect ratio.

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

**Register tokens.** Four extra learnable tokens sit next to `[CLS]`. They give the network somewhere to park global information. Without them, some patches end up with very large norms and show up as "artifact" spots in attention maps (Darcet et al., 2024). The registers make those go away.

## The problem: patch features get worse with long training

The authors watched two numbers during training:

- **ImageNet linear classification** using the `[CLS]` token (a global task);
- **Pascal VOC linear segmentation** using the patch tokens (a dense task).

Classification kept going up. Segmentation peaked around 200k iterations and then went down. For the ViT-7B it even dropped below where it was early on. To see why, they looked at cosine-similarity maps between one patch and all the others. Early in training the maps were sharp and local: a patch on a cat's ear was only similar to other cat patches. Later, lots of unrelated patches also became similar to it. So basically the patch features were losing their locality and drifting towards a global summary of the whole image. The image-level goal and the patch-level quality were pulling in opposite directions.

## The fix: Gram anchoring

Let $X_S \in \mathbb{R}^{P \times d}$ be the student's L2-normalized patch features for one image ($P$ patches, $d$ dimensions). The **Gram matrix** $X_S X_S^\top \in \mathbb{R}^{P \times P}$ holds the cosine similarity of every pair of patches. You can read it as the *structure* of the feature map: which patches go together and which don't.

DINOv3 keeps one more network around, the **Gram teacher**. It's an earlier checkpoint of the EMA teacher, from back when the patch features were still clean. With $X_G$ as its normalized patch features on the same image, the new loss is

$$
\mathcal{L}_{\text{Gram}} = \left\lVert X_S X_S^\top - X_G X_G^\top \right\rVert_F^2
$$

where $\lVert \cdot \rVert_F$ is the Frobenius norm (square root of the sum of all squared entries).

This is the part that took me a moment: the loss compares *similarities*, not the features themselves. If you forced $X_S = X_G$, the student could never get better than that old checkpoint. Matching only the Gram matrices lets the features rotate or change in any way, as long as the pairwise relationships stay the same. In plain words: "keep the same patch structure, but feel free to encode it better."

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

How they actually use it:

- The loss is added in a **refinement phase** that starts after the main pretraining (1M iterations). It's only computed on global crops.
- Every 10k iterations, the Gram teacher is refreshed by copying the current EMA teacher.
- Even though it's added late, it still **repairs** features that had already gone bad.
- There's an extra gain from giving the Gram teacher images at twice the resolution and then downsampling its feature map. Its patch features come out smoother and more detailed.

The refinement objective is

$$
\mathcal{L}_{\text{Ref}} = w_D\,\mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + w_{DK}\,\mathcal{L}_{\text{DKoLeo}} + w_{\text{Gram}}\,\mathcal{L}_{\text{Gram}}
$$

with the weights $w$ chosen by the authors. They noticed that the Gram loss makes the iBOT loss drop faster but hardly changes the DINO loss. That hints that the Gram and iBOT losses shape the features in a similar, local way, while the DINO loss does something different.

If "Gram matrix" sounds familiar, it's probably from neural style transfer (Gatys et al., 2015). There it correlates *channels* to capture texture. In DINOv3 it correlates *patches*, so it captures spatial structure instead.

## After the main training

1. **High-resolution adaptation.** A short phase with mixed, bigger crop sizes (global crops of 512 and 768 pixels), with Gram anchoring still on. This is what lets the model handle high-resolution inputs well.
2. **Distillation.** The 7B model becomes a frozen teacher for training smaller models: ViT-S, S+, B, L, H+ and ConvNeXt versions. These are the ones most people will actually use.
3. **Extras.** A text encoder aligned to the features (dino.txt) and a separate model trained on satellite images.

## Using a released model

The small model `facebook/dinov3-vits16-pretrain-lvd1689m` has 12 blocks, 6 heads, 384-dimensional tokens and a normal GELU MLP (1536 hidden units). The SwiGLU feed-forward layers are in the big 7B teacher and the "plus" variants (ViT-S+ and H+), not in this one. For a 224×224 input with patch size 16 you get $14 \times 14 = 196$ patches. The output sequence is 1 `[CLS]` + 4 registers + 196 patches = 201 tokens. The easy mistake here is forgetting the registers, so skip them when you pull out the patch tokens:

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

- DINOv3 is the DINOv2 recipe scaled up to a 7B ViT and 1.7B curated images, with constant learning-rate schedules.
- At that scale, long training makes the global features better but the patch features worse.
- Gram anchoring fixes it by matching the student's patch-to-patch similarity matrix to an earlier checkpoint's, without freezing the features themselves.
- 2D RoPE with box jittering makes the features cope better with different resolutions, and register tokens make them cleaner.
- In practice you use one of the distilled models and skip the 4 register tokens when you grab patch features.

## Resources

- Siméoni et al., "DINOv3", 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Darcet et al., "Vision Transformers Need Registers", ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Gatys, Ecker, Bethge, "A Neural Algorithm of Artistic Style", 2015. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
- Official code and weights: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Hugging Face docs: [DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3)
