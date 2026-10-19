---
title: "CNNs aren't rotation invariant: what geometric augmentation can and can't fix"
slug: cnn-rotation-and-geometric-augmentation
date: 2020-06-02
topic: Deep Learning Foundations
tags: [university-lab, cnn, data-augmentation, invariance, cifar-10]
summary: "Why a CNN trained on upright images falls apart on rotated ones, why convolution gives translation equivariance but not rotation invariance, and why augmentation only helps for the transformations you actually sample."
sources:
  - title: "732A78 Deep Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Deep Learning, Sections 7.4 (Dataset Augmentation) and 9.4 (Convolution and Pooling as an Infinitely Strong Prior)"
    author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville"
    url: "https://www.deeplearningbook.org/"
    type: book
  - title: "A survey on Image Data Augmentation for Deep Learning"
    author: "Connor Shorten, Taghi M. Khoshgoftaar"
    url: "https://doi.org/10.1186/s40537-019-0197-0"
    type: paper
  - title: "Group Equivariant Convolutional Networks"
    author: "Taco S. Cohen, Max Welling"
    url: "https://arxiv.org/abs/1602.07576"
    type: paper


status: draft
---

A small CNN I trained on CIFAR-10 got 59.5% test accuracy. Then the lab rotated every test image by 90 degrees and asked the same network again: 22.4%. Same images, same labels, just turned sideways. That drop is a good way to see what a CNN does and doesn't build in, and what augmentation can do about it.

## What convolution gives you for free

A conv layer slides the same filter over every position. So if the object moves 5 pixels to the right, the feature map moves 5 pixels to the right too. That's called **translation equivariance**: shift the input, and the output shifts the same way. Written as a formula, with $T$ a shift and $f$ a conv layer:

$$
f(T x) = T f(x)
$$

Pooling and the final classifier then make the prediction mostly insensitive to small shifts (approximately **invariant**: $g(Tx) \approx g(x)$).

Nothing like that holds for rotation. A filter that learned "horizontal edge at the top of a wheel" doesn't fire for the same edge rotated 90 degrees, because it's now a vertical edge, a different pattern. The network only knows the orientations it saw in training, and CIFAR-10 photos are almost all upright. So a sideways truck looks like nothing the network knows.

In my lab answer I explained the drop as "we rotated X without changing y and built a new model, so the coefficients have not changed properly". That's muddled. No model was rebuilt, and the labels *shouldn't* change: a rotated truck is still a truck. The real reason is the one above: the trained filters are tied to the orientations in the training data, and convolution doesn't make them rotation invariant.

## Augmentation: show the model the variations

If the network can't infer rotations, you can show them. **Data augmentation** applies random label-preserving changes to training images on the fly, so every epoch sees slightly different versions. In Keras that was:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,        # random rotation in [-30, +30] degrees
    horizontal_flip=True,     # mirror left-right
    vertical_flip=True,       # mirror top-bottom
    validation_split=0.2,
)
train_flow = datagen.flow(X_train, Y_train, batch_size=64, subset="training")
val_flow = datagen.flow(X_train, Y_train, batch_size=64, subset="validation")
# model.fit(train_flow, validation_data=val_flow, epochs=200)
```

(The current Keras way is preprocessing layers like `RandomRotation` and `RandomFlip` inside the model, but the idea is the same.)

With this, trained for 200 epochs, the results were:

| Model | Upright test images | Rotated 90° |
|---|---|---|
| No augmentation | 0.5950 | 0.2240 |
| With augmentation | 0.6295 | 0.3085 |

Better on both, but rotated accuracy is still far below upright. That makes sense once you look at what was sampled. Rotations went up to 30 degrees. A single flip is a mirror image, not a rotation, and both flips together make a 180-degree turn. So the training images covered angles near 0° and near 180° (plus mirrored versions), but nothing near 90°. Augmentation teaches invariance to the transformations you sample, and roughly their neighbours, not to everything.

Training also got slower to converge. The lab asked about this, and the reason is that the model never sees the exact same image twice, so the task is harder and you need more epochs (we went from 25 to 200).

## Choosing augmentations

The rule I took away: **augment with transformations that keep the label and that you expect at test time.**

- If test images can come in any orientation (satellite or microscope images), use rotations over the full range, including 90-degree steps.
- If they're always upright (most phone photos), a vertical flip mostly adds unrealistic upside-down cars and boats. It may cost accuracy on the upright test set for no benefit. In hindsight our `vertical_flip=True` was a questionable choice for CIFAR-10.
- Some flips change the label, like digits 6 and 9, or left/right arrows.

Augmentation isn't the only option. Test-time augmentation averages predictions over several transformed copies of each test image. Rotation-equivariant architectures (group-equivariant CNNs) build rotations into the filters, the same way ordinary convolution builds in translations.

## Key takeaways

- Convolution is translation equivariant by design, but it knows nothing about rotation. Trained filters only cover orientations seen in training.
- A 90° rotation took our CIFAR-10 CNN from 59.5% to 22.4% accuracy.
- Augmentation helps for the transformations you sample. With ±30° rotations and flips, the 90°-rotated accuracy only rose to 30.9%.
- Pick augmentations that keep the label and match test conditions. More augmentation also needs more epochs.

## Resources

- 732A78 Deep Learning, Linköping University (CNN lab).
- Goodfellow, Bengio and Courville, [Deep Learning](https://www.deeplearningbook.org/), Sections 7.4 and 9.4.
- Shorten and Khoshgoftaar, [A survey on Image Data Augmentation for Deep Learning](https://doi.org/10.1186/s40537-019-0197-0), Journal of Big Data, 2019.
- Cohen and Welling, [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576), ICML 2016.
