export const topics = [
  { key: "Computer Vision", slug: "computer-vision", blurb: "Detection, segmentation, self-supervised features and the geometry of images." },
  { key: "Geospatial & GIS", slug: "geospatial", blurb: "Photogrammetry, LiDAR, rasters and polygons: putting pixels on the map." },
  { key: "3D Vision & Point Clouds", slug: "3d", blurb: "Points, voxels, meshes and the projections between 3D and 2D." },
  { key: "Signal & Image Processing", slug: "signal", blurb: "Waves, wavelets, Fourier transforms, filters and image statistics." },
  { key: "Deep Learning Foundations", slug: "deep-learning", blurb: "Backpropagation, normalisation, losses and the plumbing of training." },
  { key: "NLP & LLMs", slug: "nlp", blurb: "From bigram models to attention and GPT, built up from scratch." },
  { key: "ML & Statistics", slug: "ml-stats", blurb: "Validation, ensembles, error analysis and the maths around models." },
  { key: "Algorithms", slug: "algorithms", blurb: "Data structures, performance and the coding tools I learned along the way." },
];
export const readingMinutes = (body: string = "") => Math.max(3, Math.round(body.replace(/```[\s\S]*?```/g, "").split(/\s+/).length / 220));
