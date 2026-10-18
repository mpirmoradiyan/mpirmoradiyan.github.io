---
title: "Waves and the Fourier spectrum: amplitude, frequency and phase"
slug: waves-and-the-fourier-spectrum
date: 2017-12-23
topic: Signal & Image Processing
tags: [waves, fourier, spectrum, phase, seismic, signal-processing]
summary: "What amplitude, period, frequency, wavelength and phase mean, and how the Fourier series and transform turn a signal into an amplitude and a phase spectrum."
sources:
  - title: "Seismic Data Processing (Investigations in Geophysics No. 2), Chapter 1: Fundamentals of signal processing"
    author: "Öz Yilmaz"
    url: "https://wiki.seg.org/wiki/Seismic_Data_Analysis"
    type: book
  - title: "Advanced Engineering Mathematics, chapter on Fourier analysis (Fourier series, integrals and transforms)"
    author: "Erwin Kreyszig"
    type: book


status: draft
---

Seismic data, sound and images are all made of waves. Before we can talk about seismic wavelets or image filters, we need a precise vocabulary for a single wave (amplitude, frequency, phase) and a tool that breaks any signal into many simple waves. That tool is Fourier analysis. I first wrote this material in Persian in 2017, while working on seismic data quality control. This is a rewritten English version.

## What a wave is

A wave is energy spread out in space that moves through it. There are three broad families: mechanical waves, electromagnetic waves and matter waves. Seismic surveys use sound waves, which are mechanical. Mechanical waves obey Newton's laws and need a material medium such as air, water or rock layers.

We can describe a wave as a function of time (what one sensor records) or as a function of space (a snapshot along a line). A seismic section is a two-dimensional wave field: one axis is time and the other is distance along the surface.

## The parameters of a simple wave

![A 4 Hz sine wave with its peak, trough, period and amplitude marked](fig-wave-parameters.png)

For a sine wave recorded over time:

- **Amplitude** $A$ is the largest displacement from zero. A positive maximum is a **peak** and a negative one is a **trough**. In seismic recording, amplitude is often a voltage.
- **Period** $T$ is the time between two neighbouring peaks (or troughs), that is, the time for one full cycle. It is measured in seconds.
- **Frequency** $f$ is the number of full cycles per second, measured in hertz (Hz). It is the inverse of the period: $f = 1/T$.
- **Phase** tells us where in its cycle the wave is at a reference time, usually $t = 0$. It is an angle: one full cycle is $360°$, or $2\pi$ radians.

If we look at the wave along a spatial axis instead:

- **Wavelength** $\lambda$ is the distance between two neighbouring peaks, in metres.
- **Wavenumber** is the spatial version of frequency: the number of cycles per unit length, $k = 1/\lambda$. Geophysicists often quote it in cycles per kilometre. Physicists often use the angular wavenumber $2\pi/\lambda$ instead, so check which one a text means.

A cosine wave with all these pieces is

$$
y(t) = A\cos(2\pi f t + \theta).
$$

The argument of the cosine has two terms. The first, $2\pi f t$, is the angle the wave has travelled through by time $t$. The second, $\theta$, is the **phase constant**: the angle at which the wave started at $t = 0$. Keep these two terms in mind. Later notes show that changing either one shifts the phase, and the two kinds of change behave differently.

## Phase as a time delay

Suppose a sine wave with period $T = 0.5$ s reaches its first peak $0.12$ s later than a reference wave that peaks at $t = 0$. We say it has a **time lag** of $0.12$ s. One full cycle is $2\pi$ radians, so the **phase lag** is

$$
\frac{0.12}{0.5} \times 2\pi = 0.48\pi \text{ rad}.
$$

By convention, **phase is the negative of the phase lag**, so this wave has phase $-0.48\pi$. A simple picture helps: put the starting time of each oscillator on a number line with zero at the origin. Starting late (to the right in time) gives a negative phase. Starting early gives a positive phase. Starting exactly at zero gives phase zero.

## A spring as a single-frequency system

Hang a weight on an ideal spring, pull it down by $0.8$ units and let go at $t = 0$. The displacement traces a cosine. With a period of $80$ ms, the frequency is $1/0.08 = 12.5$ Hz. A stiffer spring released from $0.4$ units might have a period of $40$ ms, so $25$ Hz.

Now take a second spring identical to the first, and release it at the moment the first spring passes through zero, $20$ ms later. Both have the same amplitude and frequency. The only difference is a delay of $20$ ms, which is a quarter of the period, so the phase lag is $\pi/2$ ($90°$) and the phase is $-90°$.

![Three spring motions with their amplitude and phase spectra](fig-spring-spectra.png)

If we plot each oscillation's amplitude against its frequency we get its **amplitude spectrum**. If we plot its phase against frequency we get its **phase spectrum**. The first and third springs have the same amplitude spectrum and different phase spectra. This is the key idea: the amplitude spectrum says *how much* of each frequency is present, and the phase spectrum says *when* each one is positioned in time.

## Fourier series: periodic functions as sums of sines

A function $f(x)$ is **periodic** with period $p$ if $f(x + p) = f(x)$ for all $x$. If $p$ is a period, so is $2p$, $3p$, and so on.

A function with period $2L$ can be written as a **Fourier series**:

$$
f(x) = a_0 + \sum_{n=1}^{\infty}\left[a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L}\right],
$$

with coefficients given by the Euler formulas

$$
a_0 = \frac{1}{2L}\int_{-L}^{L} f(x)\,dx,\quad
a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\frac{n\pi x}{L}\,dx,\quad
b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\frac{n\pi x}{L}\,dx.
$$

These formulas work because the sines and cosines are **orthogonal** on any interval of length $2L$: the integral of the product of any two different members of the family is zero. Multiplying $f$ by one basis function and integrating picks out exactly its own coefficient.

Using Euler's formula $e^{i\phi} = \cos\phi + i\sin\phi$, the same series has a compact complex form:

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i n\pi x/L}, \qquad
c_n = \frac{1}{2L}\int_{-L}^{L} f(x)\,e^{-i n\pi x/L}\,dx.
$$

## From series to the Fourier transform

A non-periodic function can be seen as a periodic one with an infinitely long period. As $L \to \infty$, the discrete frequencies $n\pi/L$ merge into a continuous variable $\omega$, and the sum becomes an integral. This works when $f$ is piecewise continuous, has left and right derivatives everywhere, and is absolutely integrable ($\int_{-\infty}^{\infty}|f(x)|\,dx$ exists). The result is the Fourier transform pair:

$$
\hat f(\omega) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(x)\,e^{-i\omega x}\,dx,
\qquad
f(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \hat f(\omega)\,e^{i\omega x}\,d\omega.
$$

Here $\omega$ is angular frequency and $\hat f(\omega)$ is a complex number for each $\omega$. Its magnitude is the amplitude spectrum and its angle is the phase spectrum. (Other books put the $2\pi$ factors in different places; the ideas are the same.)

## Why it is called a spectrum

Read the inverse transform from right to left. It says $f(x)$ is a superposition of oscillations $e^{i\omega x}$, one for every frequency $\omega$, each weighted by $\hat f(\omega)$. A prism does the same with white light: it splits the light into colours, and each colour is a frequency. White light is the sum of its colours. In this analogy $f$ is the white light, the Fourier transform is the prism, and $\hat f(\omega)$ is the set of colours with their strengths. That is why $\hat f$ is called the **spectral representation** of $f$.

The squared magnitude $|\hat f(\omega)|^2$ is the **spectral density**. Its integral over all frequencies is proportional to the total energy of the signal, and its integral from $a$ to $b$ tells how much the frequencies between $a$ and $b$ contribute.

The spring shows where this comes from. For a mass $m$ on a spring with constant $k$, Newton's law gives $m\ddot y + k y = 0$. Multiplying by $\dot y$ and integrating gives

$$
\tfrac{1}{2}m\dot y^2 + \tfrac{1}{2}k y^2 = E,
$$

kinetic plus potential energy equals a constant $E$. The solution is $y = A\cos(\omega_0 t + \theta)$ with $\omega_0 = \sqrt{k/m}$. Substituting gives $E = \tfrac{1}{2}kA^2$: **energy is proportional to the squared amplitude**. A more complex system whose motion is a Fourier series has one such term per frequency, so the total energy is a sum of squared amplitudes over a discrete spectrum. The Fourier transform extends this to a continuous spectrum.

```python
import numpy as np

fs = 1000                       # sampling rate (Hz)
t = np.arange(0, 2, 1/fs)   # 2 s, so 12.5 Hz falls exactly on a frequency bin
# 12.5 Hz cosine, plus a 25 Hz cosine delayed by a quarter period
y = 0.8*np.cos(2*np.pi*12.5*t) + 0.4*np.cos(2*np.pi*25*t - np.pi/2)

Y = np.fft.rfft(y) / (len(y)/2)          # scale so peaks equal amplitudes
f = np.fft.rfftfreq(len(y), 1/fs)
for k in np.argsort(np.abs(Y))[-2:]:
    print(f"{f[k]:5.1f} Hz  amplitude {abs(Y[k]):.2f}  phase {np.degrees(np.angle(Y[k])):6.1f} deg")
```

The output recovers both components: $0.80$ at $12.5$ Hz with phase $0°$ and $0.40$ at $25$ Hz with phase $-90°$.

## Key takeaways

- Amplitude, frequency and phase fully describe a single sinusoid; frequency is $1/T$, wavenumber is $1/\lambda$.
- Phase is the negative of the phase lag: a wave that starts late has negative phase.
- Fourier series write periodic signals as sums of orthogonal sines and cosines; the Fourier transform extends this to non-periodic signals.
- The magnitude of the transform is the amplitude spectrum, its angle is the phase spectrum, and the squared magnitude measures how energy is spread over frequency.

## Resources

- Öz Yilmaz, *Seismic Data Processing*, SEG Investigations in Geophysics No. 2, 1987, Chapter 1 (fundamentals of signal processing). The expanded 2001 edition, *Seismic Data Analysis*, is free on the SEG Wiki: <https://wiki.seg.org/wiki/Seismic_Data_Analysis>.
- Erwin Kreyszig, *Advanced Engineering Mathematics*, Wiley, chapter "Fourier Analysis" (Fourier series, Euler formulas, orthogonality, Fourier integral and transform, and the spectrum and energy of a spring–mass system).
