---
title: "Reading drone image metadata: EXIF, XMP, GPS, gimbal angles and ground sampling distance"
slug: drone-image-metadata-exif-xmp
date: 2024-04-18
topic: Geospatial & GIS
tags: [exif, xmp, drone, uav, metadata, gps, gsd, photogrammetry]
summary: "Where a drone photo keeps its position, altitude, camera angles and lens data (EXIF and XMP), how to read them reliably, and how to turn them into ground sampling distance, footprints and overlap."
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

A drone photo is more than pixels. The camera writes where it was, how high, which way it pointed and what lens it used into the same file. Photogrammetry software uses this as the starting guess for camera positions. Mapping tools use it to place images on a map, check coverage and compute pixel size on the ground. Getting it wrong, for example by mixing up two different altitudes, gives errors of tens of metres. This note explains where the metadata lives, how to read it, and the three calculations it is most often used for.

## Two containers: EXIF and XMP

**EXIF** is the older standard, from the camera industry. In a JPEG it sits in an APP1 segment near the start of the file as a small TIFF-style table of numbered tags. The main groups are:

- **Image and camera:** `Make`, `Model`, `DateTimeOriginal`, `ExposureTime`, `FNumber`, `ISOSpeedRatings`, `FocalLength`, `FocalLengthIn35mmFilm`, image width and height.
- **GPS:** `GPSLatitude`, `GPSLongitude`, `GPSAltitude` and their "Ref" partners.

EXIF GPS values have a quirk. Latitude is stored as three rational numbers (degrees, minutes, seconds), and the sign is stored separately as a letter in `GPSLatitudeRef` (`N` or `S`). The same holds for longitude with `E`/`W`. Forgetting the reference letter puts every image in the western or southern hemisphere on the wrong side of the planet:

```python
def dms_to_deg(dms, ref):
    """EXIF (deg, min, sec) + 'N'/'S'/'E'/'W' -> signed decimal degrees."""
    d, m, s = (float(x) for x in dms)          # rationals -> floats
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg

dms_to_deg((46, 30, 0.0), "N")    # 46.5
dms_to_deg((8, 0, 0.0), "W")      # -8.0
```

**XMP** is Adobe's newer, extensible format: an XML (RDF) packet, also stored in an APP1 segment. Anyone can add a namespace, and drone makers do. DJI cameras, for example, write a `drone-dji` namespace with fields such as `GpsLatitude`, `GpsLongitude`, `AbsoluteAltitude`, `RelativeAltitude`, `GimbalPitchDegree`, `GimbalYawDegree`, `GimbalRollDegree` and `FlightYawDegree`. Multispectral and thermal cameras add more, such as band names, irradiance and calibration values. Many of these exist only in XMP, so EXIF-only readers never see them.

Because XMP is plain XML inside the file, a small reader needs nothing but the standard library:

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

On a synthetic image this returns `{'GpsLatitude': '+46.5000000', 'RelativeAltitude': '+120.10', 'GimbalPitchDegree': '-90.00', ...}`. Values are strings and must be converted. Some writers use child elements instead of attributes, so a full reader should handle both. In practice **ExifTool** is the most complete reader: it knows thousands of tags, including maker-specific ones, and handles every file type. Its Python wrapper returns a dictionary per file with keys like `XMP:RelativeAltitude`. Lighter Python options (exifread, Pillow) are fine for standard EXIF. I tried several of these. The useful lesson was that they disagree on tag names and on what they return for missing tags. So pick one, check which tags are missing in a batch before trusting it, and validate the result (types, ranges) before using it.

## Which altitude?

A drone photo often carries three heights, and they mean different things:

- **Relative altitude**: height above the **take-off point**, from the barometer. It is not the height above the ground under the drone. Over a slope, the two differ by the change in terrain elevation:
  $$
  h_{\text{AGL}} = z_{\text{takeoff}} + h_{\text{rel}} - z_{\text{ground}}(x, y),
  $$
  where $z_{\text{takeoff}}$ and $z_{\text{ground}}$ come from a terrain model. Taking off at 92 m and flying at a relative 120 m over a hill at 130 m leaves only 82 m of clearance.
- **Absolute altitude** / `GPSAltitude`: height from GNSS. Depending on the device it is referenced to the ellipsoid or to mean sea level (the geoid). The two can differ by tens of metres, so check against a known point before you mix sources.
- **Height above ground** is what you actually need for pixel size. It usually has to be computed, as above.

## Camera orientation

Gimbal angles describe where the camera points: **pitch** ($-90°$ is straight down, or nadir; $0°$ is the horizon), **yaw** (heading, clockwise from north) and **roll**. Flight angles describe the aircraft, which is not the same thing when the gimbal stabilises the camera. For mapping, a quick check is useful: images with pitch far from $-90°$ are oblique and break nadir-only assumptions, such as simple footprints. Yaw tells you how the image is rotated relative to north.

## Ground sampling distance and footprint

The **ground sampling distance** (GSD) is the size of one pixel on the ground. By similar triangles in the pinhole camera, for a nadir image:

$$
\text{GSD} = \frac{s_w \, h}{f \, W},
$$

where $s_w$ is the sensor width (mm), $f$ the real focal length (mm), $W$ the image width in pixels and $h$ the height above ground (m). The footprint is $\text{GSD} \cdot W$ by $\text{GSD} \cdot H$. For a 1-inch sensor (13.2 mm wide, 5472 px) with an 8.8 mm lens:

| height above ground | GSD | footprint |
|---|---|---|
| 60 m | 1.6 cm/px | 90 × 60 m |
| 120 m | 3.3 cm/px | 180 × 120 m |

EXIF `FocalLength` is the real focal length, while `FocalLengthIn35mmFilm` is scaled to a full-frame sensor (43.3 mm diagonal). The ratio between them is the crop factor. If the sensor size is not in the metadata, it can be recovered from that ratio: here $8.8 \times 43.27 / 15.86 \approx 24$ mm equivalent.

To draw a footprint on a map, rotate the image rectangle by the yaw and shift it to the camera position in a metric projection (such as UTM):

```python
import numpy as np

def footprint(east, north, yaw_deg, gsd, W, H):
    hw, hh = gsd * W / 2, gsd * H / 2
    corners = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])  # image right, up
    t = np.radians(yaw_deg)                       # yaw is clockwise from north
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return corners @ R.T + [east, north]
```

**Overlap** between consecutive photos follows directly. If photos are taken every $b$ metres along the short side of the image, the forward overlap is $1 - b / (\text{GSD} \cdot H)$. At 120 m with photos every 30 m, that is $1 - 30/120 = 75\%$, a typical target for photogrammetry. Computing it from the metadata of a finished flight is a quick way to find gaps before processing.

## Other uses and one warning

Timestamps (`DateTimeOriginal`) let you sort images, match them to flight logs, or split flights. They usually carry no time zone, so store the zone separately. Camera serial numbers and lens data let you group images for calibration. And GPS tags are personal data: strip them (for example with `exiftool -all=`) before publishing photos taken near people's homes.

## Key takeaways

- EXIF holds standard camera and GPS tags. XMP is extensible XML where drone makers put altitudes, gimbal angles and sensor-specific values.
- EXIF GPS is degrees/minutes/seconds plus a hemisphere letter. Always apply the reference.
- Relative altitude is above take-off, not above ground. Absolute altitude may be ellipsoidal or orthometric.
- GSD $= s_w h / (f W)$ with height above ground. From it follow footprint and overlap.
- Use one robust reader, check for missing tags, and validate values before use.

## Resources

- CIPA DC-008, [Exif standard](https://www.cipa.jp/std/documents/download_e.html?CIPA_DC-008-2023-E) (tag definitions, GPS rationals and reference tags).
- Adobe, [XMP Specification](https://github.com/adobe/XMP-Toolkit-SDK/tree/main/docs) (RDF/XML packet and embedding in JPEG).
- Phil Harvey, [ExifTool tag names](https://exiftool.org/TagNames/), including the DJI XMP tags.
- P. R. Wolf, B. A. DeWitt, B. E. Wilkinson, *Elements of Photogrammetry with Applications in GIS*, McGraw-Hill (vertical photographs, scale and ground coverage).
