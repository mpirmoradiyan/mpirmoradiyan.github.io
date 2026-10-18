---
title: "Reading drone image metadata: EXIF, XMP, GPS, gimbal angles and ground sampling distance"
slug: drone-image-metadata-exif-xmp
date: 2024-04-18
topic: Geospatial & GIS
tags: [exif, xmp, drone, uav, metadata, gps, gsd, photogrammetry]
summary: "Where a drone photo keeps its position, height, camera angles and lens info, how to read them, and how to get pixel size on the ground, footprints and overlap from them."
sources:
  - title: "Exif Version 3.0 (CIPA DC-008)"
    author: "CIPA / JEITA"
    url: "https://www.cipa.jp/std/documents/download_e.html?CIPA_DC-008-2023-E"
    type: docs
  - title: "XMP Specification Part 1: Data model, serialization and core properties"
    author: "Adobe"
    url: "https://github.com/adobe/XMP-Toolkit-SDK/tree/main/docs"
    type: docs
  - title: "ExifTool tag names: EXIF, GPS, XMP and DJI tags"
    author: "Phil Harvey"
    url: "https://exiftool.org/TagNames/"
    type: docs
  - title: "Elements of Photogrammetry with Applications in GIS"
    author: "Wolf, DeWitt & Wilkinson"
    type: book


status: draft
---

A drone photo is more than just pixels. The camera also writes into the file where it was, how high it was, which way it was pointing and what lens it used. Photogrammetry software uses this as a first guess for the camera positions. Mapping tools use it to put images on a map, check coverage and work out the pixel size on the ground. If you get it wrong, say by mixing up two different altitudes, you can be off by tens of metres. These are my notes on where the metadata lives, how to read it, and the three calculations it's most often used for.

## Two containers: EXIF and XMP

**EXIF** is the older standard, from the camera industry. In a JPEG it sits in an APP1 segment near the start of the file, as a small TIFF-style table of numbered tags. The main groups are:

- **Image and camera:** `Make`, `Model`, `DateTimeOriginal`, `ExposureTime`, `FNumber`, `ISOSpeedRatings`, `FocalLength`, `FocalLengthIn35mmFilm`, image width and height.
- **GPS:** `GPSLatitude`, `GPSLongitude`, `GPSAltitude` and their "Ref" partners.

EXIF GPS values have one odd habit. Latitude is stored as three rational numbers (degrees, minutes, seconds), and the sign is stored somewhere else, as a letter in `GPSLatitudeRef` (`N` or `S`). Same for longitude with `E`/`W`. If you forget the letter, every image from the western or southern hemisphere ends up on the wrong side of the planet:

```python
def dms_to_deg(dms, ref):
    """EXIF (deg, min, sec) + 'N'/'S'/'E'/'W' -> signed decimal degrees."""
    d, m, s = (float(x) for x in dms)          # rationals -> floats
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg

dms_to_deg((46, 30, 0.0), "N")    # 46.5
dms_to_deg((8, 0, 0.0), "W")      # -8.0
```

**XMP** is Adobe's newer format that anyone can extend: an XML (RDF) packet, also stored in an APP1 segment. Anyone can add their own namespace, and drone makers do. DJI cameras, for example, write a `drone-dji` namespace with fields like `GpsLatitude`, `GpsLongitude`, `AbsoluteAltitude`, `RelativeAltitude`, `GimbalPitchDegree`, `GimbalYawDegree`, `GimbalRollDegree` and `FlightYawDegree`. Multispectral and thermal cameras add even more, like band names, irradiance and calibration values. A lot of these only exist in XMP, so readers that only look at EXIF never see them.

Since XMP is just plain XML sitting inside the file, a small reader only needs the standard library:

```python
import re
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"

def read_xmp(path):
    """Return the attributes of all rdf:Description elements in a file's XMP."""
    data = open(path, "rb").read()
    m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", data, re.S)
    if not m:
        return {}
    meta = {}
    for desc in ET.fromstring(m.group(0)).iter(RDF + "Description"):
        for key, value in desc.attrib.items():
            meta[key.split("}")[1]] = value      # strip the namespace URI
    return meta
```

On a synthetic image this returns `{'GpsLatitude': '+46.5000000', 'RelativeAltitude': '+120.10', 'GimbalPitchDegree': '-90.00', ...}`. The values are strings, so you have to convert them. Some writers use child elements instead of attributes, so a full reader should handle both. In practice **ExifTool** is the most complete reader. It knows thousands of tags, including maker-specific ones, and handles every file type. Its Python wrapper gives back a dictionary per file with keys like `XMP:RelativeAltitude`. Lighter Python options (exifread, Pillow) are fine for standard EXIF. I tried several of these, and what I took away was that they don't agree on tag names or on what they return when a tag is missing. So pick one, check which tags are missing across a batch before trusting it, and sanity-check the values (types, ranges) before using them.

## Which altitude?

This was the confusing part for me. A drone photo often carries three heights, and they mean different things:

- **Relative altitude**: height above the **take-off point**, from the barometer. It's not the height above the ground right under the drone. Over a slope, the two differ by how much the terrain rises or drops:
  $$
  h_{\text{AGL}} = z_{\text{takeoff}} + h_{\text{rel}} - z_{\text{ground}}(x, y),
  $$
  where $z_{\text{takeoff}}$ and $z_{\text{ground}}$ come from a terrain model. Take off at 92 m, fly at a relative 120 m over a hill at 130 m, and you only have 82 m of clearance.
- **Absolute altitude** / `GPSAltitude`: height from GNSS (satellite positioning). Depending on the device, it's measured from the ellipsoid or from mean sea level (the geoid). Those can differ by tens of metres, so check against a known point before mixing sources.
- **Height above ground** is what you really need for pixel size. Usually you have to compute it yourself, as above.

## Which way the camera points

Gimbal angles say where the camera is pointing: **pitch** ($-90°$ is straight down, called nadir; $0°$ is the horizon), **yaw** (heading, clockwise from north) and **roll**. Flight angles describe the drone itself, which isn't the same thing when the gimbal keeps the camera steady. A quick check that helps for mapping: images with pitch far from $-90°$ are oblique (tilted), and they break anything that assumes straight-down views, like simple footprints. Yaw tells you how the image is rotated relative to north.

## Ground sampling distance and footprint

The **ground sampling distance** (GSD) is simply how big one pixel is on the ground. From similar triangles in the pinhole camera, for a straight-down image:

$$
\text{GSD} = \frac{s_w \, h}{f \, W},
$$

where $s_w$ is the sensor width (mm), $f$ the real focal length (mm), $W$ the image width in pixels and $h$ the height above ground (m). The footprint is $\text{GSD} \cdot W$ by $\text{GSD} \cdot H$. For a 1-inch sensor (13.2 mm wide, 5472 px) with an 8.8 mm lens:

| height above ground | GSD | footprint |
|---|---|---|
| 60 m | 1.6 cm/px | 90 × 60 m |
| 120 m | 3.3 cm/px | 180 × 120 m |

Watch out here: EXIF `FocalLength` is the real focal length, while `FocalLengthIn35mmFilm` is scaled to a full-frame sensor (43.3 mm diagonal). The ratio between them is the crop factor. If the sensor size isn't in the metadata, you can get it back from that ratio: here $8.8 \times 43.27 / 15.86 \approx 24$ mm equivalent.

To draw a footprint on a map, rotate the image rectangle by the yaw and move it to the camera position in a metric projection (like UTM):

```python
import numpy as np

def footprint(east, north, yaw_deg, gsd, W, H):
    hw, hh = gsd * W / 2, gsd * H / 2
    corners = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])  # image right, up
    t = np.radians(yaw_deg)                       # yaw is clockwise from north
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return corners @ R.T + [east, north]
```

**Overlap** between photos in a row falls out of this directly. If photos are taken every $b$ metres along the short side of the image, the forward overlap is $1 - b / (\text{GSD} \cdot H)$. At 120 m with a photo every 30 m, that's $1 - 30/120 = 75\%$, a common target for photogrammetry. Working it out from the metadata of a finished flight is a quick way to spot gaps before processing.

## Other uses, and one warning

Timestamps (`DateTimeOriginal`) let you sort images, match them to flight logs, or split flights. They usually have no time zone, so store the zone separately. Camera serial numbers and lens data let you group images for calibration. And GPS tags are personal data: strip them (for example with `exiftool -all=`) before posting photos taken near people's homes.

## Key takeaways

- EXIF holds the standard camera and GPS tags. XMP is extensible XML where drone makers put altitudes, gimbal angles and sensor-specific values.
- EXIF GPS is degrees/minutes/seconds plus a hemisphere letter. Always apply the letter.
- Relative altitude is above take-off, not above ground. Absolute altitude may be from the ellipsoid or from sea level.
- GSD $= s_w h / (f W)$ with height above ground. Footprint and overlap follow from it.
- Stick to one reliable reader, check for missing tags, and sanity-check values before using them.

## Resources

- CIPA DC-008, [Exif standard](https://www.cipa.jp/std/documents/download_e.html?CIPA_DC-008-2023-E) (tag definitions, GPS rationals and reference tags).
- Adobe, [XMP Specification](https://github.com/adobe/XMP-Toolkit-SDK/tree/main/docs) (RDF/XML packet and embedding in JPEG).
- Phil Harvey, [ExifTool tag names](https://exiftool.org/TagNames/), including the DJI XMP tags.
- P. R. Wolf, B. A. DeWitt, B. E. Wilkinson, *Elements of Photogrammetry with Applications in GIS*, McGraw-Hill (vertical photographs, scale and ground coverage).
