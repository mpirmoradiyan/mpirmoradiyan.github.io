---
title: "Airborne LiDAR basics: returns, classes, ground filtering and DTM, DSM and CHM"
slug: airborne-lidar-dtm-dsm-chm
date: 2024-01-17
topic: Geospatial & GIS
tags: [lidar, las, point-cloud, dtm, dsm, chm, ground-filtering, rasterization, forestry]
summary: "What an airborne LiDAR point cloud contains (returns, classes, scaled coordinates), how ground points are found, and how the three standard elevation rasters (DTM, DSM and the canopy height model) are built without pits and holes."
sources:
  - title: "LAS Specification 1.4 R15"
    author: "ASPRS"
    url: "https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf"
    type: docs
  - title: "The lidR package book"
    author: "Roussel, Goodbody & Tompalski"
    url: "https://r-lidar.github.io/lidRbook/"
    type: book
  - title: "Generating DEM from LIDAR data with progressive TIN densification"
    author: "Axelsson (2000)"
    type: paper
  - title: "A progressive morphological filter for removing nonground measurements from airborne LIDAR data"
    author: "Zhang, Chen, Whitman, Shyu, Yan & Zhang (2003)"
    type: paper
  - title: "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation"
    author: "Zhang, Qi, Wan, Wang, Xie, Wang & Yan (2016)"
    url: "https://www.mdpi.com/2072-4292/8/6/501"
    type: paper
  - title: "Generating pit-free canopy height models from airborne lidar"
    author: "Khosravipour, Skidmore, Isenburg, Wang & Hussin (2014)"
    type: paper


status: draft
---

An airborne laser scanner fires hundreds of thousands of pulses per second from a plane or drone and times their echoes. The result is a cloud of millions of 3D points over the ground, buildings and trees. Most users never look at the points directly. They use three rasters made from them: the bare-earth terrain (DTM), the top surface (DSM), and the difference between the two, which in a forest is the canopy height model (CHM). This note explains what is in a LiDAR file, how ground points are separated from the rest, and how to build the three rasters without the usual artefacts. All examples use a synthetic 60 m × 60 m forest tile.

## What a pulse records

A laser pulse is not a thin line. Its footprint on the canopy is a few centimetres to decimetres wide. Part of the beam hits a leaf and reflects, while the rest continues down through gaps and may reflect again from a branch and finally from the ground. Each echo the sensor detects is a **return**, and every return becomes a point with:

- **Return number** and **number of returns**: "2 of 3" means the second of three echoes from one pulse. First returns describe the top surface. Last returns are the best chance of reaching the ground. A "1 of 1" return from open ground and a "1 of 1" from a dense roof look the same.
- **Intensity**: the strength of the echo. It depends on the material, but also on range, angle and how the footprint was split among returns, so it needs normalising before you compare it across flights.
- **Classification**: a code per point. The ASPRS LAS standard fixes the main ones: 1 unclassified, 2 ground, 3/4/5 low/medium/high vegetation, 6 building, 7 low noise, 9 water, 18 high noise. National datasets are often delivered with at least ground and noise already classified.
- **Scan angle, GPS time, flight line** and, in newer formats, **colour** and **near-infrared**.

Coordinates in LAS/LAZ are stored as scaled integers, $x = X \cdot s + o$, where $X$ is the stored integer, $s$ the scale (such as 0.01 m) and $o$ an offset in the header. Library readers return the real value. The raw integer fields are only useful for compact processing, and mixing the two is a classic bug.

A useful first step on any tile is to drop what is clearly not surface: noise classes 7 and 18, points flagged as withheld, and heights far below the terrain. Many tools can apply these filters while reading.

## Finding the ground

If the ground is not classified, you must classify it. Ground filters all rely on one idea: the terrain is the lowest surface, and it is smooth compared with the objects on it. The classic families are:

- **Progressive morphological filters** (Zhang et al., 2003). Take the lowest point per cell, then apply a morphological opening (erosion then dilation) with a window that grows step by step. Objects narrower than the window disappear. A point is non-ground if it lies more than a height threshold above the opened surface. The threshold grows with window size, so that real slopes survive.
- **Progressive TIN densification** (Axelsson, 2000). Start from a few seed points that are surely ground, the lowest in large cells. Build a triangulated surface (TIN) through them. Then add points whose distance to the nearest triangle, and the angles they make with its corners, are small enough. Repeat until no more points qualify. This is the method behind many commercial tools.
- **Cloth simulation filter** (Zhang et al., 2016). Turn the cloud upside down and drop a simulated cloth on it. A stiff cloth rests on the (inverted) terrain without sinking into the "holes" left by trees and buildings. Points close to the cloth are ground. Cloth rigidity is the main parameter and roughly controls how steep a slope may be.

All three trade off the same errors. If they are too strict, ridges and cliff edges are cut off. If they are too loose, low bushes and car roofs end up in the terrain.

## DTM, DSM and CHM

**DTM** (digital terrain model). Interpolate the ground points onto a grid. Triangulation (TIN) with linear interpolation is the standard choice. Inverse distance weighting and kriging also work. Under dense canopy, ground points are sparse, so the DTM there is an interpolation across gaps of several metres. In my synthetic tile, with 40% of canopy pulses reaching the ground, the TIN error was about 3 cm.

**DSM** (digital surface model). For each cell, keep the highest first return. That is a z-buffer from above, the same rule as in [projecting point clouds to images](/notes/projecting-point-clouds-to-images/):

```python
import numpy as np

def p2r_max(x, y, z, res, x0, y_top, shape, subcircle=0.0):
    """Highest z per cell. With subcircle > 0, each point is replaced by
    8 points on a small circle, so it can fill neighbouring cells."""
    if subcircle > 0:
        a = np.arange(8) * np.pi / 4
        x = (x[:, None] + subcircle * np.cos(a)).ravel()
        y = (y[:, None] + subcircle * np.sin(a)).ravel()
        z = np.repeat(z, 8)
    col = np.floor((x - x0) / res).astype(int)
    row = np.floor((y_top - y) / res).astype(int)        # north up
    ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    out = np.full(shape, -np.inf)
    np.maximum.at(out, (row[ok], col[ok]), z[ok])        # keep the maximum per cell
    out[np.isinf(out)] = np.nan                          # empty cells
    return out
```

**CHM** (canopy height model) is $\text{CHM} = \text{DSM} - \text{DTM}$: the height of whatever stands on the ground. An equivalent route is to **normalise** the points first, subtracting the terrain height under each point ($z' = z - \text{DTM}(x, y)$), and then take the highest normalised point per cell. Normalised points are also what you need for per-plot height statistics such as height percentiles or canopy cover. (How those heights turn into forest parameters is the subject of [forest structure from above](/notes/forest-structure-from-above/).)

The same subtraction over a city gives a normalised DSM (nDSM) of buildings.

![Four rasters of a synthetic forest tile. The DTM is a smooth surface of hills and valleys. The DSM shows round tree crowns on top of the terrain. The CHM made from single points has scattered low pixels, or pits, inside the crowns. The CHM made with a 0.25 m subcircle has fewer pits and slightly wider crowns.](fig-dtm-dsm-chm.png)

## Pits, holes and the subcircle trick

Two artefacts appear when the cell size is close to the point spacing.

**Empty cells.** With about 4 pulses per m² and 0.5 m cells, the average cell gets one first return, so many get none. In my tile 37% of DSM cells were empty before interpolation.

**Pits.** Some first returns slip through a gap in the crown and come from deep inside it. If such a point is the only one in its cell, the cell shows a hole several metres deep in the middle of a tree. Pits break tree-top detection and crown segmentation, which look for local maxima and smooth crown shapes.

The **subcircle** trick, which lidR offers for its point-to-raster method, replaces each return by a small ring of points of radius $r$. Every return can then reach the neighbouring cells, and since the maximum wins, the high returns cover the low ones. On the synthetic tile:

| subcircle radius | empty DSM cells | pit cells inside crowns |
|---|---|---|
| 0 (single points) | 37% | 20% |
| 0.15 m | 9% | 11% |
| 0.25 m | 3% | 7% |

The price is that each crown grows by about $r$ in every direction, so keep $r$ smaller than half a cell or so. Other fixes are a median or closing filter on the CHM, or the **pit-free** algorithm of Khosravipour et al. (2014). That method builds several CHMs from first returns above increasing height thresholds, with long triangle edges removed, and takes the maximum of all of them. Whatever you choose, interpolate the remaining empty cells, for example with a TIN, rather than leaving them as zeros.

## Choosing the resolution

Pick the cell size from the point density. A cell should hold several first returns on average, so for density $\rho$ points/m², a cell side of at least $\sqrt{3/\rho}$ m is a reasonable start (about 0.9 m at 4 points/m², 0.25 m at 50 points/m²). Coarser cells need no tricks but blur individual crowns. Finer cells need subcircles or pit filling.

## Key takeaways

- Every return carries a return number, number of returns, intensity and a class. First returns describe the top surface, last returns the best chance of ground.
- Ground filters assume the terrain is the lowest smooth surface. Morphological, TIN-densification and cloth-simulation methods are the standard families.
- DTM is interpolated ground, DSM is the highest return per cell, and CHM = DSM − DTM (or the highest height-normalised return per cell).
- Cells smaller than the point spacing give empty cells and pits. The subcircle trick, filters or pit-free CHMs fix them at a small cost in crown width.

## Resources

- ASPRS, [LAS Specification 1.4 R15](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf): point record fields, return numbers and standard classes.
- J.-R. Roussel, T. Goodbody, P. Tompalski, [*The lidR package*](https://r-lidar.github.io/lidRbook/), chapters on ground classification, DTM, CHM (including the subcircle option and pit-free CHM) and height normalisation.
- P. Axelsson, "Generating DEM from LIDAR data with progressive TIN densification", *International Archives of Photogrammetry and Remote Sensing*, 33(B4), 2000.
- K. Zhang et al., "A progressive morphological filter for removing nonground measurements from airborne LIDAR data", *IEEE Transactions on Geoscience and Remote Sensing*, 41(4), 2003.
- W. Zhang et al., ["An easy-to-use airborne LiDAR data filtering method based on cloth simulation"](https://www.mdpi.com/2072-4292/8/6/501), *Remote Sensing*, 8(6), 2016.
- A. Khosravipour et al., "Generating pit-free canopy height models from airborne lidar", *Photogrammetric Engineering & Remote Sensing*, 80(9), 2014.
