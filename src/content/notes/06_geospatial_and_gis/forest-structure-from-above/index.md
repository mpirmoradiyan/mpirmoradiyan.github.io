---
title: "Forest structure from above: from tree heights and crowns to DBH, basal area and Lorey's height"
slug: forest-structure-from-above
date: 2025-09-19
topic: Geospatial & GIS
tags: [forestry, lidar, photogrammetry, allometry, dbh, basal-area, remote-sensing]
summary: "What the standard forest numbers (DBH, basal area, stems per hectare, DGV, Lorey's height, dominant height) mean, how to compute them right, and how published models guess DBH and biomass from the tree height and crown size you can see from above."
sources:
  - title: "Estimating basal area and stem volume for individual trees from lidar data"
    author: "Chen, Gong, Baldocchi & Tian (2007)"
    url: "https://pdfs.semanticscholar.org/c96a/b9d760f92cda213e188374ddf9408de85091.pdf"
    type: paper
  - title: "Predicting diameter at breast height from total height and crown length"
    author: "Cao & Dean (2013)"
    url: "https://research.fs.usda.gov/treesearch/43592"
    type: paper
  - title: "Improving AGB estimations by integrating tree height and crown radius from multisource remote sensing"
    author: "Liu, Dong, Li, Li, Wang, Mao & Deng (2024)"
    url: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0311642"
    type: paper
  - title: "Using Tree Height, Crown Area and Stand-Level Parameters to Estimate Tree Diameter, Volume, and Biomass of Pinus radiata, Eucalyptus globulus and Eucalyptus nitens"
    author: "Gonzalez-Benecke, Fernández, Gayoso, Pincheira & Wightman (2022)"
    url: "https://www.mdpi.com/1999-4907/13/12/2043"
    type: paper
  - title: "Raster Skogliga grunddata: produktbeskrivning"
    author: "Skogsstyrelsen (Swedish Forest Agency)"
    url: "https://skogsstyrelsen.se/globalassets/sjalvservice/karttjanster/skogliga-grunddata/raster-skogliga-grunddata---produktbeskrivning.pdf"
    type: docs
  - title: "Perfecting a stand-density index for even-aged forests"
    author: "Reineke (1933)"
    type: paper


status: draft
---

A forester describes a stand with a handful of numbers: how thick the trees are, how many there are per hectare, how much ground their stems cover, and how tall they are. Most of these are defined at breast height, 1.3 m above the ground. And that's exactly the part of the tree a drone or an airborne laser scanner can't see. From above, you get tree height and crown size. So here I collected the standard definitions, fixed a few formulas that are easy to get wrong, and looked at published models that try to bridge the gap between "what we can see" and "what foresters need".

## The basic tree measurement: DBH

**DBH** (diameter at breast height, written $d$ or $D_{1.3}$) is the stem's diameter 1.3 m above the ground, measured in centimetres with a calliper or a diameter tape. Pretty much everything else in forest measurement is built on it: basal area, volume, biomass and growth.

The other per-tree values are simpler. **Height** $h$ is from the ground to the tip. **Crown base height** (CBH) is the height of the lowest live branch, so **crown length** is $CL = h - \text{CBH}$ and **crown ratio** is $CR = CL / h$, a number between 0 and 1. **Crown width** is how wide the crown spreads, usually the average of two measurements at right angles.

## Stand numbers, done carefully

**Basal area** $g$ of one tree is the area of its stem cut across at 1.3 m. With $d$ in centimetres, the radius in metres is $d/200$, so

$$
g = \pi \left(\frac{d}{200}\right)^2 = \frac{\pi}{40\,000}\, d^2 \approx 0.00007854\, d^2 \quad [\text{m}^2].
$$

The stand's **basal area** $G$ is the sum over the trees in a plot, divided by the plot area $A$ in hectares:

$$
G = \frac{1}{A}\sum_{i=1}^{n} g_i \quad [\text{m}^2/\text{ha}].
$$

It's easy to forget the division by $A$. Without it you get square metres per plot, not per hectare.

**Stems per hectare** is $N = n / A$ on a fixed-area plot. Angle-count (relascope) sampling works differently, and this took me a moment: every counted tree stands for $\text{BAF}/g_i$ stems per hectare, where BAF is the basal area factor of the gauge. So $G = \text{BAF} \times (\text{count})$ and $N = \sum_i \text{BAF}/g_i$. Thin trees count for many stems, thick trees for few.

**Mean diameters.** Three are in common use, and they're not the same number:

- the **arithmetic mean** $\bar d = \frac{1}{n}\sum d_i$;
- the **quadratic mean diameter** $D_q = \sqrt{\frac{1}{n}\sum d_i^2}$, the diameter of the tree with average basal area;
- the **basal-area-weighted mean diameter**, called **DGV** in Swedish forest data (*grundytevägd medeldiameter*):

$$
\text{DGV} = \frac{\sum g_i d_i}{\sum g_i} = \frac{\sum d_i^3}{\sum d_i^2}.
$$

Each tree's diameter is weighted by its basal area, so big trees count more. You'll sometimes see a shortcut written as $0.00007854 \sum d_i^3 / G$. That's only right if $G$ is the plot sum of $g_i$, not the per-hectare value. The ratio $\sum d^3 / \sum d^2$ is safer.

**Heights.** The plain mean height $\bar h$ gets dragged down by small trees stuck under the canopy. **Lorey's height** weights each tree by its basal area, just like DGV:

$$
H_L = \frac{\sum g_i h_i}{\sum g_i} = \frac{\sum d_i^2 h_i}{\sum d_i^2}.
$$

Swedish data call it **HGV** (*grundytevägd medelhöjd*). It follows stand volume better than $\bar h$. **Dominant height** $H_{\text{dom}}$ is the mean height of the 100 thickest trees per hectare. On a plot of $A$ hectares that means the $100A$ thickest trees, so 10 trees on a 0.1 ha plot. Dominant height barely changes when a stand is thinned, which is why it's used for **site index**: the dominant height a stand reaches at a reference age. It's often modelled with a Chapman–Richards curve $H_{\text{dom}} = a(1 - e^{-bt})^c$, where $t$ is stand age, $a$ the height it levels off at, $b$ a rate and $c$ a shape parameter.

**Reineke's stand density index** compares the stem count to that of a stand with a reference mean diameter of 25 cm:

$$
\text{SDI} = N \left(\frac{D_q}{25}\right)^{1.605}.
$$

The exponent is **positive**. A version with $-1.605$ floats around in some notes, but it makes a stand of big trees look less dense, which is the opposite of the whole point.

Here's all of it on a synthetic plot:

```python
import numpy as np

rng = np.random.default_rng(0)
A = 0.1                                       # plot area, ha
d = rng.gamma(shape=9, scale=2.6, size=60)    # DBH, cm
h = 1.3 + (d / (1.6 + 0.20 * d)) ** 2         # Näslund-type height curve, m

g   = np.pi * (d / 200) ** 2                  # basal area per tree, m²
G   = g.sum() / A                             # m²/ha
N   = len(d) / A                              # stems/ha
Dq  = np.sqrt((d ** 2).mean())
DGV = (g * d).sum() / g.sum()                 # = Σd³ / Σd²
H_L = (g * h).sum() / g.sum()                 # Lorey's height
H_dom = h[np.argsort(d)[-round(100 * A):]].mean()
SDI = N * (Dq / 25) ** 1.605
# N=600, G=27.5, mean d=23.1, Dq=24.2, DGV=27.2, mean h=14.7, H_L=15.9, H_dom=17.7
```

It's always $\bar d \le D_q \le \text{DGV}$, and the gap gets bigger the more the diameters vary. The height model above is **Näslund's function** $h = 1.3 + d^2/(a + b d)^2$, the classic height–diameter curve in Nordic forestry.

![Left: histogram of DBH in a synthetic plot with the arithmetic mean, quadratic mean and basal-area-weighted mean marked; DGV is the largest. Right: predicted versus true DBH for a height-only model and a height plus crown area model; the second scatter is tighter around the 1:1 line.](fig-diameters-and-dbh-models.png)

## Bridging the gap: guessing DBH from height and crown

From a canopy height model and a crown segmentation you get, per tree, the height $H$ and the crown area $CA$ (or radius $R$). Allometry is the idea that parts of a tree grow in proportion to each other, roughly following power laws. So a natural model is

$$
\text{DBH} = b_0\,(H - 1.3)^{b_1}\, CA^{b_2},
$$

which turns into a straight-line model once you take logs. Four published studies show the same lesson from different angles.

**Chen et al. (2007)**, blue oak woodland in California, small-footprint LiDAR. They compared lots of models for single-tree basal area and stem volume. A linear DBH model from maximum height and crown area, $\text{DBH} = 2.29 + 1.6\,H_{100} + 0.26\,C_a$ (DBH in cm, $H_{100}$ the highest return in m, $C_a$ the crown area in m²), followed by $B = \pi(\text{DBH}/2)^2$, was among the best when crowns were segmented correctly. Their own new idea was the **canopy geometric volume** $V_c$, the volume under the crown surface. They modelled basal area and stem volume as simple power functions of $V_c$, and these held up best when the segmentation made mistakes.

**Cao & Dean (2013)** start from the **uniform stress theory**. The idea is that a stem tapers so that wind pushing on the crown causes about the same bending stress all along the stem. So the diameter at some height depends on the leaf area above it times the lever arm to the centre of that leaf area. With $H_{MC} = H - CL/2$ (height of the crown centre) and the lever arm $L = H_{MC} - 1.37$ m, they fit $D = b_1 (A_L \cdot L)^{b_2}$, where the leaf area $A_L$ is itself estimated from height and crown length. Both inputs come from LiDAR. Models with crown length clearly beat height-only power models, and adding stems per hectare helped all of them.

**Gonzalez-Benecke et al. (2022)**, plantations of *Pinus radiata*, *Eucalyptus globulus* and *E. nitens*, built three levels of models for DBH (cm), stem volume and above-stump biomass: height only; height plus crown area; and height, crown area and stand variables like age and stems per hectare. Height-only models worked, but their errors grew with tree size. Adding crown area gave the biggest single improvement and got rid of that size bias.

**Liu et al. (2024)** skip DBH and predict above-ground biomass $W$ directly from height $H$ and crown radius $R$ measured from UAV data. The reference biomass came from standard $W = a(D^2H)^b$ equations. They tried nine forms. Models that used both $H$ and $R$ beat models with just one of them, and the quadratic $W = a(H+R)^2 + b(H+R) + c$ was the most consistent across forest types (in mixed coniferous stands, $R^2 \approx 0.93$).

On synthetic trees where crown size carries real information, the effect is easy to see:

```python
rng = np.random.default_rng(1)
H  = rng.uniform(8, 30, 300)                                   # tree height, m
CA = np.exp(rng.normal(np.log(0.12 * H**1.5), 0.35))           # crown area, m²
DBH = 1.2 * (H - 1.3)**0.75 * CA**0.30 * np.exp(rng.normal(0, 0.08, 300))  # cm

def fit_power(y, *xs):
    """Fit y = b0 * x1^b1 * x2^b2 ... by least squares on logs."""
    X = np.column_stack([np.ones_like(y)] + [np.log(x) for x in xs])
    b, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    return b, np.exp(X @ b)

b1, pred1 = fit_power(DBH, H - 1.3)          # height only     -> RMSE 2.9 cm
b2, pred2 = fit_power(DBH, H - 1.3, CA)      # + crown area    -> RMSE 1.8 cm
```

Two things to watch out for. First, turning a log-scale fit back with `exp` gives you the median, not the mean, so it comes out a bit low. A correction factor like $e^{\sigma^2/2}$ fixes most of that. Second, every coefficient is **specific to a species and region**. A model fitted to plantation eucalyptus tells you nothing about Swedish spruce until you refit it on local field plots.

## From trees to stand values

The literature has two broad routes. In the **individual-tree approach** (for example Hyyppä et al., 2001), each detected tree gets a predicted DBH from models like the ones above. The stand numbers then come from summing $g_i$, $d_i^3$ or $d_i^2 h_i$ over the trees in an area and dividing by that area. Its weak spots are trees hidden under the canopy, which never get detected, and crowns that get merged or split, which mess up both the tree count and the crown areas (see [polygon validity and shape metrics](/notes/polygon-validity-and-shape-metrics/) for checking crown shapes). In the **area-based approach** (Næsset, 2002), stand values like $G$, $N$ or volume are regressed straight on height statistics of the point cloud over field plots, with no tree detection at all. Hidden trees don't hurt it much, but you don't get a list of trees. When volume per tree is needed, it's usually a species function $v = a\, d^b h^c$.

## Key takeaways

- Basal area per tree is $\pi (d/200)^2$ m² with $d$ in cm. Divide the plot sum by plot area to get m²/ha.
- $\bar d$, $D_q$ and DGV are three different means. DGV and Lorey's height both weight trees by basal area ($\sum d^3/\sum d^2$ and $\sum d^2 h/\sum d^2$).
- Dominant height uses the 100 thickest trees **per hectare**. Reineke's SDI uses a **positive** exponent of 1.605.
- Height alone predicts DBH and biomass with errors that grow for big trees. Adding crown size (area, radius or length) fixes a lot of that.
- Allometric coefficients are local: refit them with field data before trusting them.

## Resources

- Chen, Q., Gong, P., Baldocchi, D. & Tian, Y. Q. (2007). [Estimating basal area and stem volume for individual trees from lidar data](https://pdfs.semanticscholar.org/c96a/b9d760f92cda213e188374ddf9408de85091.pdf). *Photogrammetric Engineering & Remote Sensing*, 73(12), 1355–1365.
- Cao, Q. V. & Dean, T. J. (2013). [Predicting diameter at breast height from total height and crown length](https://research.fs.usda.gov/treesearch/43592). In *Proceedings of the 15th Biennial Southern Silvicultural Research Conference*, e-Gen. Tech. Rep. SRS-175, USDA Forest Service, pp. 201–205.
- Gonzalez-Benecke, C. A., Fernández, M. P., Gayoso, J., Pincheira, M. & Wightman, M. G. (2022). [Using tree height, crown area and stand-level parameters to estimate tree diameter, volume, and biomass of *Pinus radiata*, *Eucalyptus globulus* and *Eucalyptus nitens*](https://www.mdpi.com/1999-4907/13/12/2043). *Forests*, 13(12), 2043.
- Liu, X., Dong, L., Li, S., Li, Z., Wang, Y., Mao, Z. & Deng, L. (2024). [Improving AGB estimations by integrating tree height and crown radius from multisource remote sensing](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0311642). *PLOS ONE*, 19(10), e0311642.
- Skogsstyrelsen. [Raster Skogliga grunddata: produktbeskrivning](https://skogsstyrelsen.se/globalassets/sjalvservice/karttjanster/skogliga-grunddata/raster-skogliga-grunddata---produktbeskrivning.pdf): Swedish definitions of DGV, HGV and related stand variables.
- Reineke, L. H. (1933). Perfecting a stand-density index for even-aged forests. *Journal of Agricultural Research*, 46, 627–638.
- Hyyppä, J., Kelle, O., Lehikoinen, M. & Inkinen, M. (2001). A segmentation-based method to retrieve stem volume estimates from 3-D tree height models produced by laser scanners. *IEEE Transactions on Geoscience and Remote Sensing*, 39(5), 969–975.
- Næsset, E. (2002). Predicting forest stand characteristics with airborne scanning laser using a practical two-stage procedure and field data. *Remote Sensing of Environment*, 80(1), 88–99.
- West, P. W. (2015). *Tree and Forest Measurement*, 3rd ed., Springer: textbook reference for basal area, angle-count sampling, mean diameters and stand heights.
