---
title: "Segmentation decoders: FCN skip fusion, U-Net, and how to upsample without checkerboards"
slug: segmentation-decoders-fcn-and-unet
date: 2022-11-09
topic: Computer Vision
tags: [semantic-segmentation, fcn, u-net, upsampling, transposed-convolution, pixel-shuffle]
summary: "How a segmentation network gets from a stride-32 feature map back to a full-size label map, how FCN-style score adding differs from U-Net-style feature concatenation, and why transposed convolutions and pixel shuffle can leave checkerboard patterns."
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

A classification backbone squeezes a 512×512 image down to a 16×16 grid of rich features. But a segmentation model has to give a label to every one of the 262,144 pixels. The part that gets you back up there is the **decoder**. I built a few of them in Keras on top of ImageNet backbones: an FCN-8s, an FCN-4s, and a U-Net that used pixel shuffle for upsampling. These are my notes on the two main decoder designs, what my "FCN" really turned out to be, and how the choice of upsampling layer changes the output. How to pull multi-scale features out of the backbone is in [Tapping a CNN backbone](/notes/cnn-backbone-feature-extraction/).

## The problem: deep features are coarse

Each stage of a backbone halves the resolution. After five stages the feature map has stride 32, so there's one feature vector for each 32×32 block of pixels. These features know *what* is in the block, but they've lost *where* exactly the edges are. Earlier stages (stride 4 or 8) still know where things are, but not so much what they are. Every decoder is basically a way to combine the two.

## FCN: add up class scores from several depths

Long, Shelhamer and Darrell's Fully Convolutional Network (2015) turned a classifier into a per-pixel predictor by swapping its fully connected layers for convolutions. Upsampling the stride-32 class scores straight to full size (**FCN-32s**) gave blobby masks. Their fix was to add predictions from shallower layers:

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

With VGG-sized features (256, 512 and 512 channels) and five classes, this head has only about 6,400 parameters. The mixing happens in **score space**: each depth casts a vote per class, and the votes get added up. That's cheap, but the shallow layers can only nudge the class scores in a linear way. They can't really reason about the deep features. The original paper also used learnable transposed convolutions (started off as bilinear) instead of fixed bilinear resizing.

## What I actually built: concatenation, which is U-Net's idea

This one surprised me when I looked back at my code. The model I'd called FCN-8s was doing something else. At each step it upsampled the deep *features* (not class scores) with bilinear resizing, **concatenated** them with the backbone features at the next stride, and ran two 3×3 convolutions with 512 or 256 filters. Only at the very end did a 1×1 convolution with softmax turn things into class scores. FCN-4s just added one more step, down to stride 4.

So that's not FCN's score fusion at all. It's the U-Net pattern, used on only three skip levels. And the difference matters:

- **Concatenate then convolve** lets the decoder learn non-linear mixes of "what" (deep) and "where" (shallow) features. It can express a lot more.
- It also costs a lot more. The four-block U-Net-style decoder below has about 4 million parameters, compared to 6,400 for the FCN head.

## U-Net: a mirror-image decoder with skips at every stride

Ronneberger, Fischer and Brox's U-Net (2015) takes this all the way. The decoder mirrors the encoder: at every stride it upsamples, concatenates the encoder features at the same resolution (that's the **skip connection**), and applies two 3×3 convolutions. It was made for biomedical images, where you have few training samples and very thin structures, so getting the edges exactly right matters most.

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

Some things I picked up while building these on pretrained backbones:

- **Take the skip before the pooling.** For VGG, the useful skip at each stride is the last conv of the block (`block4_conv4`), not the pooled output (`block4_pool`), which already belongs to the next stride. Mix them up and you get shape errors, or the resolution quietly halves.
- **Resize to the skip's size, not by a fixed factor.** If the input sides aren't multiples of 32, you get odd sizes somewhere, and a fixed ×2 won't line up.
- **Think about batch norm on the skips and in the decoder.** One of my FCN-4s versions normalised the skip features before concatenating, because pretrained encoder features and freshly started decoder features can be on very different scales.

For comparison, the [Feature Pyramid Network](/notes/feature-pyramid-networks/) goes back to addition (after 1×1 projections to a common width) and is made for detection at every level, not for one full-resolution output.

## Picking the upsampling layer

The decoder needs some way to go from $H \times W$ to $2H \times 2W$. There are three common options.

**Fixed interpolation, then convolution.** A bilinear (or nearest) resize, then a learned 3×3 convolution. The upsampling itself has nothing to learn, but it's smooth and stable. My FCN versions used this.

**Transposed convolution** (often called "deconvolution"). The way I picture it: each input pixel stamps a learned $k \times k$ kernel onto the output, shifting by the stride $s$ each time. Where stamps overlap, they add up. If $k$ isn't divisible by $s$, some output pixels get more stamps than others. A 3×3 kernel with stride 2 and all weights equal to 1, applied to a constant input, gives:

```text
1 2 1 2
2 4 2 4
1 2 1 2
2 4 2 4
```

That's a **checkerboard artefact**, and it comes straight from the layer's geometry. A 4×4 kernel with stride 2 gives a flat output for this all-ones kernel, but with learned, uneven weights it can still make the pattern. Odena, Dumoulin and Olah explain this really well and suggest resize-then-convolve instead.

**Pixel shuffle** (sub-pixel convolution, Shi et al. 2016). A convolution makes $C \cdot r^2$ channels at low resolution, and then a reshuffle moves each group of $r^2$ channels into an $r \times r$ block of output pixels. My U-Net used this: a 3×3 convolution to $4C$ channels, then `depth_to_space` with factor 2. It's efficient and fully learnable. The catch: with random initialisation, the four sub-pixel positions come from four separate filters, so even a flat input gives a repeating 2×2 pattern.

The fix is **ICNR initialisation** (Aitken et al. 2017): start the $r^2$ filters of each group as copies of one random filter. Then, at the start, the layer acts exactly like nearest-neighbour upsampling followed by a convolution, and it only drifts away from that if it helps.

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

This tripped me up: the order of the copies has to match how `pixel_shuffle` groups channels. In PyTorch, output channel $c$ takes input channels $c r^2, \dots, c r^2 + r^2 - 1$, so copies that sit next to each other (`repeat_interleave`) are the right choice.

![Four untrained upsampling layers applied to a constant input. A 3 by 3 stride-2 transposed convolution shows vertical stripes and a checkerboard, randomly initialised pixel shuffle shows a 2 by 2 grid pattern, while pixel shuffle with ICNR initialisation and bilinear resize followed by convolution both give a flat grey output.](fig-checkerboard.png)

Training can shrink these artefacts, but the network wastes effort doing it, and faint grids often stay in the output probabilities. In segmentation, where the output is an argmax over classes, they show up as jagged or striped edges.

## Key takeaways

- A segmentation decoder mixes coarse deep features (they know "what") with sharp shallow ones (they know "where").
- FCN adds up **class scores** from several depths; it's tiny, but it can only correct the deep prediction linearly.
- U-Net **concatenates features** at every stride and runs convs on them; it can express much more and costs much more. Lots of models called "FCN" are really U-Net-style.
- Take skips before pooling, resize to the skip's exact size, and think about batch norm in a decoder on top of a pretrained encoder.
- To avoid checkerboards, go for resize-then-convolve, or pixel shuffle with ICNR initialisation, instead of plain transposed convolutions.

## Resources

- Long, J., Shelhamer, E. and Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR. https://arxiv.org/abs/1411.4038
- Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. https://arxiv.org/abs/1505.04597
- Odena, A., Dumoulin, V. and Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. https://distill.pub/2016/deconv-checkerboard/
- Shi, W. et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*. CVPR. https://arxiv.org/abs/1609.05158
- Aitken, A. et al. (2017). *Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize*. https://arxiv.org/abs/1707.02937
- Related notes: [Tapping a CNN backbone](/notes/cnn-backbone-feature-extraction/), [Feature Pyramid Networks](/notes/feature-pyramid-networks/), [Panoptic segmentation](/notes/panoptic-segmentation/).
