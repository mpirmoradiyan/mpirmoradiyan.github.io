---
title: "Tapping a CNN backbone: multi-scale feature maps from VGG and ResNet"
slug: cnn-backbone-feature-extraction
date: 2025-03-25
topic: Computer Vision
tags: [pytorch, torchvision, feature-extraction, vgg, resnet, feature-pyramid, backbone]
summary: "How to pull feature maps out of a pretrained VGG19, VGG19-BN or ResNet-50 at several depths with torchvision's create_feature_extractor, how to read their shapes and strides, and why detectors and segmenters need them."
sources:
  - title: "Feature extraction for model inspection (torchvision.models.feature_extraction)"
    author: "PyTorch / torchvision"
    url: "https://pytorch.org/vision/stable/feature_extraction.html"
    type: docs
  - title: "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)"
    author: "Karen Simonyan, Andrew Zisserman"
    url: "https://arxiv.org/abs/1409.1556"
    type: paper
  - title: "Deep Residual Learning for Image Recognition (ResNet)"
    author: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
    url: "https://arxiv.org/abs/1512.03385"
    type: paper
  - title: "Feature Pyramid Networks for Object Detection"
    author: "Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie"
    url: "https://arxiv.org/abs/1612.03144"
    type: paper
  - title: "Fully Convolutional Networks for Semantic Segmentation"
    author: "Jonathan Long, Evan Shelhamer, Trevor Darrell"
    url: "https://arxiv.org/abs/1411.4038"
    type: paper
  - title: "Build the Neural Network (Learn the Basics)"
    author: "PyTorch Tutorials"
    url: "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"
    type: docs
  - title: "pytorch-summary (torchsummary)"
    author: "Shubham Chandel (sksq96)"
    url: "https://github.com/sksq96/pytorch-summary"
    type: repo


status: draft
---

A classifier only cares about the last layer of a network. A detector or a segmenter needs a lot more: feature maps from several depths, so it can find small objects on fine grids and big objects on coarse ones. You almost never write that backbone yourself. You take a pretrained CNN and tap into it at the right spots. These are my notes on doing that for VGG19, VGG19-BN and ResNet-50 with torchvision, and what to check along the way. (The notebook starts with the official PyTorch "Build the Neural Network" tutorial as a warm-up. The experiments below go past it.)

## Why several depths?

As an image goes through a CNN, two things happen at the same time:

- **Resolution goes down.** Every pooling layer or stride-2 convolution halves the height and width.
- **Meaning goes up.** Early layers react to edges and textures in small neighborhoods. Deep layers cover big parts of the image and react to object parts and whole objects.

How much a feature map has been shrunk is called its **stride** $s$. For an input of size $H \times W$, a map at stride $s$ has size $\frac{H}{s} \times \frac{W}{s}$. With my 512×512 test input:

| Stride | Map size | What it is good for |
|---|---|---|
| 4 | 128×128 | fine boundaries, tiny objects |
| 8 | 64×64 | small objects |
| 16 | 32×32 | medium objects |
| 32 | 16×16 | large objects, global context |

No single map is good at everything. Feature Pyramid Networks (FPN) deal with this by taking the maps at strides 4 to 32 and adding a top-down path that carries the "meaning" from the deep maps back into the high-resolution ones. Mask R-CNN, RetinaNet, Panoptic FPN and lots of segmenters all start from a set of maps like this. So the first practical step is always the same: get those maps out of a backbone.

## Look at the backbone first

Before picking where to tap, look at how the shapes change. `torchsummary` prints every layer's output shape and parameter count:

```python
import torch
from torchsummary import summary
from torchvision.models import vgg19

device = "cuda" if torch.cuda.is_available() else "cpu"
model = vgg19().to(device)
summary(model, input_size=(3, 512, 512), device=device)
```

For VGG19 you see five blocks of 3×3 convolutions, each ending in a max-pool that halves the size: 512 → 256 → 128 → 64 → 32 → 16. Meanwhile the channel count grows from 64 to 512.

## `create_feature_extractor`: name the nodes you want

torchvision can trace a model with `torch.fx` and build a new module that hands back any intermediate results you ask for, in a dictionary. First list the node names, then pick the ones you want:

```python
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

model = vgg19(weights=VGG19_Weights.DEFAULT).eval()
train_nodes, eval_nodes = get_graph_node_names(model)   # every tappable node

body = create_feature_extractor(model, return_nodes={
    "features.18": "stride8",    # 3rd max-pool
    "features.27": "stride16",   # 4th max-pool
    "features.36": "stride32",   # 5th max-pool
})

with torch.no_grad():
    feats = body(torch.randn(1, 3, 512, 512))
for name, f in feats.items():
    print(name, tuple(f.shape))
# stride8  (1, 256, 64, 64)
# stride16 (1, 512, 32, 32)
# stride32 (1, 512, 16, 16)
```

The keys of `return_nodes` are the graph node names. The values are whatever names you want in the output dictionary. At first I labeled them by spatial size ("64", "32", "16"), but naming them by stride is clearer, because the stride stays the same when the input size changes.

The extractor only runs the graph up to the deepest node you asked for, so VGG's classifier layers don't run at all.

## VGG19-BN: same network, different indices

`vgg19_bn` adds a BatchNorm after every convolution. So each conv now takes three entries in `features` (conv, BN, ReLU) instead of two, and the pooling layers move to indices 26, 39 and 52:

```python
from torchvision.models import vgg19_bn

model = vgg19_bn(weights="DEFAULT").eval()
body = create_feature_extractor(model, return_nodes={
    "features.26": "stride8", "features.39": "stride16", "features.52": "stride32"})
# same shapes as plain VGG19: (256, 64, 64), (512, 32, 32), (512, 16, 16)
```

Lesson learned: don't reuse node indices between model variants. Look them up with `get_graph_node_names` every time.

## ResNet-50 and the reused-ReLU trap

For ResNet-50 I first picked nodes like `layer3.0.relu` and `layer4.0.relu`. The output was:

```text
layer3.0.relu    -> (1, 256, 64, 64)
layer4.0.relu    -> (1, 512, 32, 32)
layer4.2.relu_2  -> (1, 2048, 16, 16)
```

The last one is right, but the first two aren't the stage outputs I wanted. Stage 3 of ResNet-50 should give 1024 channels at stride 16. This one confused me for a bit. The reason is how torchvision writes a `Bottleneck` block: it defines one `self.relu` module and **calls it three times** (after each of the three convolutions). FX names the three calls `relu`, `relu_1` and `relu_2`. So `layer3.0.relu` is the activation after the *first* 1×1 convolution of the first block in stage 3. That convolution cuts the channels to 256, and the stride-2 convolution after it hasn't run yet, so the map is still 64×64.

Only `relu_2` of the *last* block in a stage is the real stage output. The simplest and safest thing is to tap the stage modules themselves:

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
body = create_feature_extractor(model, return_nodes={
    "layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"})

with torch.no_grad():
    feats = body(torch.randn(1, 3, 512, 512))
# c2 (1, 256, 128, 128)   stride 4
# c3 (1, 512, 64, 64)     stride 8
# c4 (1, 1024, 32, 32)    stride 16
# c5 (1, 2048, 16, 16)    stride 32
```

The names C2 to C5 come from the FPN paper, where $C_i$ is the output of stage $i$ at stride $2^i$.

## A quick check for any backbone

After building an extractor, push a dummy input through and check that each map has the size you expect:

```python
def check_strides(feats, input_size=512):
    for name, f in feats.items():
        s = input_size // f.shape[-1]
        print(f"{name:10s} channels={f.shape[1]:5d}  size={f.shape[-1]:4d}  stride={s}")
```

If two maps that were supposed to be different stages have the same stride, or a channel count looks off, you've probably tapped a node inside a block instead of at its end.

## How this connects to real models

- **Segmentation.** `torchvision.models.segmentation.fcn_resnet50` uses exactly this kind of ResNet body. Its last stages are changed to keep a higher resolution (dilated convolutions), and a small convolutional head on top predicts a class for every pixel.
- **Detection.** torchvision's detection models wrap a ResNet with a similar "return these layers" mechanism and feed C2–C5 into an FPN.
- **Your own heads.** Once you have the dictionary of maps, adding an FPN (`torchvision.ops.FeaturePyramidNetwork`) or a U-Net-style decoder only takes a few lines.

## Key takeaways

- Detectors and segmenters need feature maps at several strides (usually 4 to 32), not just the final output.
- `get_graph_node_names` lists the nodes you can tap. `create_feature_extractor` gives back the ones you pick as a dictionary.
- Node indices change between variants (VGG19 vs VGG19-BN), so always look them up.
- In torchvision ResNets one ReLU module gets called three times per block. Tap `layer1` to `layer4` to get clean stage outputs.
- Always run a dummy input and check channels and strides before building a head on top.

## Resources

- torchvision docs, "Feature extraction for model inspection": [pytorch.org/vision](https://pytorch.org/vision/stable/feature_extraction.html)
- Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Long, Shelhamer, Darrell, "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
- PyTorch tutorial used as the notebook's warm-up, "Build the Neural Network": [pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- torchsummary: [sksq96/pytorch-summary](https://github.com/sksq96/pytorch-summary)
