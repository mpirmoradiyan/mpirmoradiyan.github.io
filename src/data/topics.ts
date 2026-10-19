export const topics = [
  { key: "Computer Vision", slug: "computer-vision", blurb: "Detection, segmentation, self-supervised features and the geometry of images." },
  { key: "Geospatial & GIS", slug: "geospatial", blurb: "Photogrammetry, LiDAR, rasters and polygons: putting pixels on the map." },
  { key: "3D Vision & Point Clouds", slug: "3d", blurb: "Points, voxels, meshes and the projections between 3D and 2D." },
  { key: "Signal & Image Processing", slug: "signal", blurb: "Waves, wavelets, Fourier transforms, filters and image statistics." },
  { key: "Deep Learning Foundations", slug: "deep-learning", blurb: "Backpropagation, normalisation, losses and the plumbing of training." },
  { key: "NLP & LLMs", slug: "nlp", blurb: "From bigram models to attention and GPT, built up from scratch." },
  { key: "Machine Learning", slug: "machine-learning", blurb: "Clustering, trees and forests, SVMs, ensembles and the first steps of reinforcement learning." },
  { key: "Model Evaluation", slug: "evaluation", blurb: "Honest validation, thresholds, imbalance and reading errors beyond one number." },
  { key: "Statistical Inference", slug: "inference", blurb: "Bootstrap, permutation tests, multivariate tests, multiple testing and missing data." },
  { key: "Bayesian Methods", slug: "bayesian", blurb: "Priors, posteriors, credible intervals, Bayesian networks and Gaussian processes." },
  { key: "Computational Statistics", slug: "computational-stats", blurb: "MCMC, random number generation, optimisation, EM and the numerics underneath." },
  { key: "Time Series & State Space", slug: "time-series", blurb: "AR models, Kalman filters and smoothers, particle filters and hidden Markov models." },
  { key: "Programming & Data", slug: "programming", blurb: "Spark and SQL, algorithms, floating-point numbers and making code fast." },
];
export const readingMinutes = (body: string = "") => Math.max(3, Math.round(body.replace(/```[\s\S]*?```/g, "").split(/\s+/).length / 220));
