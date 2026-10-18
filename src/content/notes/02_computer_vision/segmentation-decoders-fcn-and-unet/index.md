---
title: "Segmentation decoders: FCN skip fusion, U-Net, and how to upsample without checkerboards"
slug: segmentation-decoders-fcn-and-unet
date: 2022-11-09
topic: Computer Vision
tags: [semantic-segmentation, fcn, u-net, upsampling, transposed-convolution, pixel-shuffle]
summary: "How a segmentation network turns a stride-32 feature map back into a full-resolution label map, the difference between FCN-style score fusion and U-Net-style feature concatenation, and why transposed convolutions and pixel shuffle can produce checkerboard artefacts."
sources:
  - title: "Fully Convolutional Networks for Semantic Segmentation"
    author: "Jonathan Long, Evan Shelhamer, Trevor Darrell (CVPR 2015)"
    url: "https://arxiv.org/abs/1411.4038"
    type: paper
  - title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    author: "Olaf Ronneberger, Philipp Fischer, Thomas Brox (MICCAI 2015)"
    url: "https://arxiv.org/abs/1505.04597"
    type: paper
  - title: "Deconvolution and Checkerboard Artifacts"
    author: "Augustus Odena, Vincent Dumoulin, Chris Olah (Distill, 2016)"
    url: "https://distill.pub/2016/deconv-checkerboard/"
    type: paper
  - title: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    author: "Wenzhe Shi et al. (CVPR 2016)"
    url: "https://arxiv.org/abs/1609.05158"
    type: paper
  - title: "Checkerboard artifact free sub-pixel convolution"
    author: "Andrew Aitken et al. (2017)"
    url: "https://arxiv.org/abs/1707.02937"
    type: paper


status: draft
---

A classification backbone shrinks a 512×512 image to a 16×16 grid of rich features. A segmentation model has to give one label to each of the 262,144 pixels. The part that does this is the **decoder**. I built several of them in Keras on top of ImageNet backbones: an FCN-8s and an FCN-4s, and a U-Net whose upsampling used pixel shuffle. This note explains the two main decoder designs, what my "FCN" actually was, and how the choice of upsampling layer affects the output. How to tap the backbone for its multi-scale features is covered in [Tapping a CNN backbone](/notes/cnn-backbone-feature-extraction/).

## The problem: deep features are coarse

Each stage of a backbone halves the resolution. After five stages the feature map has stride 32: one feature vector per 32×32 block of pixels. These features know *what* is in the block but have lost *where* exactly the boundaries are. Earlier stages (stride 4 or 8) still have sharp locations but weak semantics. Every decoder is a way to combine the two.

## FCN: add class scores from several depths

Long, Shelhamer and Darrell's Fully Convolutional Network (2015) made a classifier dense by replacing its fully connected layers with convolutions. Upsampling the stride-32 class scores straight to full size (**FCN-32s**) gave blobby masks. Their fix was to add predictions from shallower layers:

1. A 1×1 convolution turns the stride-32 features into class scores.
2. Upsample them ×2 and **add** class scores computed (again with a 1×1 convolution) from the stride-16 features: this is **FCN-16s**.
3. Upsample ×2 again and add scores from the stride-8 features: **FCN-8s**.
4. Upsample ×8 to full resolution.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FCN8sHead(nn.Module):
    def __init__(self, c3, c4, c5, n_classes):
        super().__init__()
        self.score3 = nn.Conv2d(c3, n_classes, 1)   # stride 8
        self.score4 = nn.Conv2d(c4, n_classes, 1)   # stride 16
        self.score5 = nn.Conv2d(c5, n_classes, 1)   # stride 32

    def forward(self, f3, f4, f5, out_size):
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        s = self.score5(f5)
        s = up(s, f4) + self.score4(f4)      # 32 -> 16
        s = up(s, f3) + self.score3(f3)      # 16 -> 8
        return F.interpolate(s, size=out_size, mode="bilinear", align_corners=False)
```

With VGG-sized features (256, 512 and 512 channels) and five classes, this head has only about 6,400 parameters. Fusion happens in **score space**: each depth casts a vote per class, and the votes are summed. That is cheap, but the shallow layers can only contribute a linear correction to the class scores. They cannot reason about the deep features. The original paper also used learnable transposed convolutions (initialised as bilinear) instead of fixed bilinear resizing.

## What I actually built: concatenation, which is U-Net's idea

Looking back at my code, the model I called FCN-8s did something different. At each step it upsampled the deep *features* (not class scores) with bilinear resizing, **concatenated** them with the backbone features at the next stride, and ran two 3×3 convolutions with 512 or 256 filters. Only at the very end did a 1×1 convolution with softmax produce class scores. FCN-4s added one more step down to stride 4.

That is not FCN's score fusion. It is the U-Net pattern, applied to only three skip levels. The difference matters:

- **Concatenate then convolve** lets the decoder learn non-linear combinations of "what" (deep) and "where" (shallow) features. It is much more expressive.
- It also costs much more. My four-block U-Net-style decoder below has about 4 million parameters, against 6,400 for the FCN head.

## U-Net: a symmetric decoder with skips at every stride

Ronneberger, Fischer and Brox's U-Net (2015) goes all the way. The decoder mirrors the encoder: at every stride it upsamples, concatenates the encoder features of the same resolution (the **skip connection**), and applies two 3×3 convolutions. It was designed for biomedical images with few training samples and very thin structures, where boundary precision matters most.

```python
class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        # resize to the skip's exact size; this also handles odd input sizes
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

# stride 32 -> 16 -> 8 -> 4 -> 2 for a 512x512 input
# shapes: (256, 32, 32), (128, 64, 64), (64, 128, 128), (64, 256, 256)
```

A few practical notes from building these on pretrained backbones:

- **Take the skip before the pooling.** For VGG, the useful skip at each stride is the last convolution of the block (`block4_conv4`), not the pooled output (`block4_pool`), which already belongs to the next stride. Mixing them up gives shape mismatches or silently halves the resolution.
- **Resize to the skip's size, not by a fixed factor.** Inputs whose sides are not multiples of 32 produce odd sizes somewhere, and a fixed ×2 then fails to match.
- **Consider batch norm on the skips and in the decoder.** One of my FCN-4s variants normalised the skip features before concatenating them, because pretrained encoder features and freshly initialised decoder features can have very different scales.

For comparison, the [Feature Pyramid Network](/notes/feature-pyramid-networks/) uses addition again (after 1×1 projections to a common width) and is built for detection at every level, not for one full-resolution output.

## Choosing the upsampling layer

The decoder needs some way to go from $H \times W$ to $2H \times 2W$. There are three common choices.

**Fixed interpolation, then convolution.** Bilinear (or nearest) resize, followed by a learned 3×3 convolution. It has no learnable upsampling weights, but it is smooth and stable. This is what my FCN variants used.

**Transposed convolution** (often called "deconvolution"). Each input pixel stamps a learned $k \times k$ kernel onto the output, moved by the stride $s$ each time. Where stamps overlap, they add. If $k$ is not divisible by $s$, some output pixels receive more stamps than others. A 3×3 kernel with stride 2 and all weights equal to 1, applied to a constant input, gives:

```text
1 2 1 2
2 4 2 4
1 2 1 2
2 4 2 4
```

That is a **checkerboard artefact** built into the layer's geometry. A 4×4 kernel with stride 2 gives a flat output for this all-ones kernel, but with learned, unequal weights it can still produce the pattern. Odena, Dumoulin and Olah describe this in detail and recommend resize-then-convolve instead.

**Pixel shuffle** (sub-pixel convolution, Shi et al. 2016). A convolution produces $C \cdot r^2$ channels at low resolution, and a reshuffle moves each group of $r^2$ channels into an $r \times r$ block of output pixels. This was what my U-Net used: a 3×3 convolution to $4C$ channels followed by `depth_to_space` with factor 2. It is efficient and fully learnable, but with random initialisation the four sub-pixel positions come from four independent filters, so even a flat input produces a 2×2 periodic pattern.

The fix is **ICNR initialisation** (Aitken et al. 2017): initialise the $r^2$ filters of each group as copies of one random filter. At the start the layer then behaves exactly like nearest-neighbour upsampling followed by a convolution, and it learns to deviate from that only if it helps.

```python
def icnr_(weight, r=2):
    out_c, in_c, kh, kw = weight.shape
    sub = torch.empty(out_c // r**2, in_c, kh, kw)
    nn.init.kaiming_normal_(sub)
    weight.data.copy_(sub.repeat_interleave(r**2, dim=0))   # r*r identical copies per output channel

conv = nn.Conv2d(16, 1 * 4, 3, padding=1, bias=False)
icnr_(conv.weight)
y = F.pixel_shuffle(conv(torch.ones(1, 16, 16, 16)), 2)   # flat output, as it should be
```

The order of copies must match how `pixel_shuffle` groups channels: in PyTorch, output channel $c$ takes input channels $c r^2, \dots, c r^2 + r^2 - 1$, so consecutive copies (`repeat_interleave`) are correct.

![Four untrained upsampling layers applied to a constant input. A 3 by 3 stride-2 transposed convolution shows vertical stripes and a checkerboard, randomly initialised pixel shuffle shows a 2 by 2 grid pattern, while pixel shuffle with ICNR initialisation and bilinear resize followed by convolution both give a flat grey output.](fig-checkerboard.png)

Training can reduce these artefacts, but it spends capacity doing so, and faint grids often survive in the output probabilities. For segmentation, where the output is an argmax over classes, they show up as jagged or striped boundaries.

## Key takeaways

- A segmentation decoder combines coarse, semantic deep features with sharp, shallow ones.
- FCN fuses **class scores** from several depths by addition; it is tiny but only linearly corrects the deep prediction.
- U-Net **concatenates features** at every stride and convolves them; it is far more expressive and more expensive. Many models called "FCN" in practice are really U-Net-style.
- Take skips before pooling, resize to the skip's exact size, and consider batch norm in a decoder on top of a pretrained encoder.
- Prefer resize-then-convolve or pixel shuffle with ICNR initialisation over plain transposed convolutions to avoid checkerboard artefacts.

## Resources

- Long, J., Shelhamer, E. and Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR. https://arxiv.org/abs/1411.4038
- Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. https://arxiv.org/abs/1505.04597
- Odena, A., Dumoulin, V. and Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. https://distill.pub/2016/deconv-checkerboard/
- Shi, W. et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*. CVPR. https://arxiv.org/abs/1609.05158
- Aitken, A. et al. (2017). *Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize*. https://arxiv.org/abs/1707.02937
- Related notes: [Tapping a CNN backbone](/notes/cnn-backbone-feature-extraction/), [Feature Pyramid Networks](/notes/feature-pyramid-networks/), [Panoptic segmentation](/notes/panoptic-segmentation/).
