export const profile = {
  name: "Mohsen Pirmoradiyan",
  role: "AI Lead & Computer Vision Engineer",
  location: "Linköping, Sweden",
  email: "mohsenpirmoradian@gmail.com",
  github: "https://github.com/mpirmoradiyan",
  linkedin: "https://www.linkedin.com/in/mohsen-pirmoradiyan",
  tagline: "I teach machines to read the forest from above. Before that, I spent fourteen years reading the earth beneath it.",
};

export const chapters = [
  {
    period: "2005 – 2019",
    title: "Reading the earth",
    place: "Seismic field operations, NISOC",
    text: "Fourteen years on land seismic crews, from junior to senior QC supervisor: making sure every sensor recording was clean, consistent and in the right place before anyone built on it.",
    carries: "Signals, sensors, geometry, and a habit of checking data before trusting it.",
  },
  {
    period: "2019 – 2021",
    title: "Learning the language",
    place: "MSc Statistics & Machine Learning, Linköping University",
    text: "A second master's degree to turn field intuition into modelling skill, finished with a thesis on anomaly detection and causal inference in 5G networks together with Ericsson Research.",
    carries: "Statistics, machine learning, and causal thinking.",
  },
  {
    period: "2022 – today",
    title: "Reading the forest from above",
    place: "Arboair AB, Linköping",
    text: "Computer vision for drone and aerial imagery of forests: I designed and lead the company's computer-vision platform and the models that map tree species and forest damage.",
    carries: "Deep learning, geospatial ML, and leading AI work end to end.",
  },
];

export const highlights = [
  "Designed and lead the company's computer-vision platform: one system from data ingestion and training data to model training, geospatial evaluation and GPU inference in production.",
  "Train transformer-based segmentation models for tree species and forest-damage mapping (Mask2Former with DINOv3 and Swin backbones, Panoptic FPN, Mask R-CNN, UNet++).",
  "Built self-supervised pretraining on unlabelled aerial imagery and semi-supervised pseudo-labelling to make the most of scarce annotations.",
  "Turn pixel predictions into GIS-ready maps: tiled inference over very large GeoTIFFs, vectorisation, geometry repair and certainty scoring.",
  "Built the orthomosaic pipeline from drone photos: LiDAR terrain models, georeferencing from drone metadata, seamline mosaicking and true orthophotos.",
  "Led the tree stem-diameter estimation project from field reference data to a production model with safe staged releases.",
];
