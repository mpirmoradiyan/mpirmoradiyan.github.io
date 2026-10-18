---
title: "Minimum, maximum and mixed phase wavelets, and why bandwidth sets resolution"
slug: minimum-phase-wavelets-and-bandwidth
date: 2017-12-23
topic: Signal & Image Processing
tags: [wavelets, seismic, minimum-phase, bandwidth, resolution, signal-processing]
summary: "Why wavelets with the same amplitude spectrum can release their energy early or late, and why a wider frequency band gives a shorter wavelet that resolves thinner layers."
sources:
  - title: "Seismic Data Processing (Investigations in Geophysics No. 2), Chapter 1: Fundamentals of signal processing"
    author: "Öz Yilmaz"
    url: "https://wiki.seg.org/wiki/Seismic_Data_Analysis"
    type: book
  - title: "Geophysical Signal Analysis"
    author: "Enders A. Robinson and Sven Treitel"
    type: book


status: draft
---

Two wavelets can contain exactly the same frequencies with the same strengths and still look very different, because their phase spectra differ. In seismic processing, the most important phase distinction is between minimum, maximum and mixed phase, since it decides whether a wavelet's energy arrives at its start or at its end. A second practical question is how many frequencies a wavelet contains, because that sets the thinnest layer we can see. This note, rewritten from my 2017 Persian notes, covers both. It follows [Seismic wavelets: amplitude and phase spectra](/notes/wavelet-amplitude-and-phase-spectra/).

## Where is the energy?

Take three wavelets with the same amplitude spectrum but different phase spectra. In the first, the biggest swing is near the start. In the second it is in the middle. In the third it is near the end. Their energy is concentrated in different places in time.

- A **minimum-phase** wavelet has its energy concentrated at the start.
- A **maximum-phase** wavelet has its energy concentrated at the end.
- Anything in between is **mixed phase**.

A minimum-phase wavelet is also **causal**: it is zero for $t < 0$. A causal system responds only after it is excited, which is how physical sources behave.

## A three-sample example

Here are four short wavelets, each with three samples:

| Wavelet | Samples | Cumulative energy after sample 0, 1, 2 |
|---|---|---|
| A | (4, 0, −1) | 16, 16, 17 |
| B | (2, 3, −2) | 4, 13, 17 |
| C | (−2, 3, 2) | 4, 13, 17 |
| D | (−1, 0, 4) | 1, 1, 17 |

The cumulative energy is the running sum of squared amplitudes. All four have total energy 17, and, less obviously, all four have the **same amplitude spectrum**. You can check this with z-transforms: $A(z) = 4 - z^2 = (2-z)(2+z)$, $B(z) = (2-z)(1+2z)$, $C(z) = (2z-1)(z+2)$ and $D(z) = (2z-1)(2z+1)$. Each wavelet is built from the same two "dipoles" or their time-reversed versions, and reversing a dipole does not change its magnitude on the unit circle.

![Left: the four three-sample wavelets. Middle: their cumulative energy. Right: their unwrapped phase spectra](fig-energy-delay.png)

What differs is how fast the energy builds up. Wavelet A reaches almost all its energy at the first sample: it has the **least energy delay**. Wavelet D gets almost nothing until the last sample: it has the **largest energy delay**. B and C sit in between and are identical in energy build-up. Among all wavelets with the same amplitude spectrum, the one with least energy delay is called **minimum delay**, and the one with most is **maximum delay**.

This is Robinson's energy-delay theorem: **the minimum-phase wavelet is the one with the least energy delay**. Since a time delay is a phase lag, the two names describe the same thing. The right panel shows it in the phase spectra: A's phase changes least across frequency (minimum phase), D's changes most (maximum phase), and B and C are mixed.

```python
import numpy as np

wavelets = {"A": [4, 0, -1], "B": [2, 3, -2], "C": [-2, 3, 2], "D": [-1, 0, 4]}
for name, w in wavelets.items():
    amp = np.abs(np.fft.fft(w, 8))           # amplitude spectrum
    energy = np.cumsum(np.square(w))         # energy build-up
    roots = np.roots(w[::-1])                # zeros of w0 + w1 z + w2 z^2
    kind = ("minimum" if np.all(np.abs(roots) > 1) else
            "maximum" if np.all(np.abs(roots) < 1) else "mixed")
    print(name, np.round(amp, 2), energy, kind)
```

All four rows print the same amplitude spectrum. A is reported as minimum phase (both zeros of its polynomial lie outside the unit circle, using the $z$ convention above), D as maximum phase, and B and C as mixed.

## Wavelets in seismic acquisition

Every seismic source makes its own wavelet. That wavelet travels into the earth, reflects from layer boundaries, and what we see on a seismic section is the result. On land the two most common sources are:

- **Dynamite**, which produces a **minimum-phase** wavelet: a sudden, causal pulse.
- **Vibroseis** (a vibrator truck sweeping through frequencies). After the recorded data are cross-correlated with the sweep, the effective wavelet is **zero phase** (often called the Klauder wavelet).

Knowing which one you have matters for later steps such as deconvolution, which usually assumes a minimum-phase wavelet.

## Bandwidth and resolution

A wavelet is a sum of frequency components. What happens when we add more of them?

![Zero-phase wavelets built from 1 to 10 Hz, 1 to 20 Hz, 1 to 60 Hz and 1 to 250 Hz](fig-bandwidth.png)

The wider the band of frequencies, the shorter the wavelet. With 1–10 Hz the central lobe is about 100 ms wide; with 1–250 Hz it is a few milliseconds, almost a single spike. This is a general rule of Fourier analysis: duration in time and width in frequency are inversely related.

This matters directly for geology. Two reflections from the top and bottom of a thin layer arrive close together in time. If the wavelet is long, the two copies overlap and blur into one event. A shorter wavelet keeps them apart. So:

- A **narrow-band** wavelet cannot resolve thin layers or small faults.
- A **broad-band** wavelet gives more detail about how the layers change.
- In the limit, a wavelet containing all frequencies equally is a **spike** (a unit impulse). It is the ideal wavelet for resolution, which is why much of seismic processing tries to widen the bandwidth and compress the wavelet towards a spike.

The low end of the band matters too. Removing low frequencies adds strong side lobes (ringing) around the main peak, which can be mistaken for extra layers.

## Key takeaways

- Wavelets with the same amplitude spectrum can differ in phase, and phase decides where the energy sits in time.
- Minimum phase means least energy delay (energy at the start, causal); maximum phase means energy at the end; everything else is mixed.
- Dynamite gives a minimum-phase wavelet; correlated Vibroseis data give a zero-phase wavelet.
- Wider bandwidth means a shorter wavelet and better vertical resolution; the ideal limit is a spike.

## Resources

- Öz Yilmaz, *Seismic Data Processing*, SEG Investigations in Geophysics No. 2, 1987, Chapter 1 (minimum, maximum and mixed phase; the four three-sample wavelets; bandwidth and resolution). The 2001 edition, *Seismic Data Analysis*, is free on the SEG Wiki: <https://wiki.seg.org/wiki/Seismic_Data_Analysis>.
- Enders A. Robinson and Sven Treitel, *Geophysical Signal Analysis*, Prentice-Hall, 1980 (the energy-delay theorem and minimum-delay wavelets).
