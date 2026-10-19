export type Stop = {
  years: string; title: string; org: string; place: string; url?: string; slug?: string;
  image: string; imageAlt: string; credit?: string; story: string; kind: "school" | "study" | "work";
};

export const journey: Stop[] = [
  {
    years: "1999", kind: "school", title: "High school diploma, Mathematics & Physics", org: "High school", place: "Isfahan, Iran",
    image: "/images/journey/isfahan-sio-se-pol.jpg", imageAlt: "Si-o-se Pol bridge in Isfahan at dusk",
    credit: "Photo: Reza Haji-pour, CC BY 3.0, via Wikimedia Commons",
    story: "Where it started: mathematics and physics in Isfahan.",
  },
  {
    years: "2000", kind: "school", title: "Pre-university, Mathematics", org: "Pre-university centre", place: "Isfahan, Iran",
    image: "/images/journey/isfahan-naqsh-e-jahan.jpg", imageAlt: "Naqsh-e Jahan Square in Isfahan",
    credit: "Photo: Arad, CC BY-SA 3.0, via Wikimedia Commons",
    story: "One more year of mathematics before the national university entrance exam.",
  },
  {
    years: "2000 – 2004", kind: "study", title: "BSc Petroleum Exploration Engineering", slug: "bsc-petroleum-exploration", org: "Petroleum University of Technology", place: "Ahvaz, Iran",
    url: "https://www.put.ac.ir", image: "/images/journey/ahvaz-white-bridge.jpg", imageAlt: "The White Bridge over the Karun river in Ahvaz",
    credit: "Photo: Alireza Javaheri, CC BY 3.0, via Wikimedia Commons",
    story: "Geology, geophysics and the physics of waves travelling through rock.",
  },
  {
    years: "2005 – 2019", kind: "work", title: "Seismic Data Acquisition QC Supervisor, junior to senior", slug: "nisoc", org: "National Iranian South Oil Company (NISOC)", place: "Khuzestan, Iran",
    url: "https://www.nisoc.ir", image: "/images/journey/nisoc-field-collage.jpg", imageAlt: "Collage of my field photos: vibroseis trucks at sunset, recording-system QC monitors, a geophone array and surveying in sand dunes",
    story: "Fourteen years in the field: designing and quality-checking 2D and 3D land seismic surveys, where every recording had to be clean before it was worth anything.",
  },
  {
    years: "2009 – 2012", kind: "study", title: "MSc Geophysics (Seismology)", slug: "msc-geophysics", org: "Institute of Geophysics, University of Tehran", place: "Tehran, Iran",
    url: "https://ig.ut.ac.ir", image: "/images/journey/ut-msc-impedance-section.jpg", imageAlt: "Illustration of an acoustic-impedance section with seismic traces, two wells and a low-impedance reservoir lens on an anticline",
    story: "A master's degree alongside field work. My thesis used prestack seismic inversion to characterise a petroleum reservoir: turning recorded waves into rock properties.",
  },
  {
    years: "2019 – 2021", kind: "study", title: "MSc Statistics & Machine Learning", slug: "msc-statistics-ml", org: "Linköping University", place: "Linköping, Sweden",
    url: "https://liu.se", image: "/images/journey/liu-campus-valla-aerial.jpg", imageAlt: "Aerial view of Campus Valla at Linköping University",
    credit: "Photo: Erik And30, CC BY-SA 4.0, via Wikimedia Commons",
    story: "A move to Sweden and a new language for the same curiosity: statistics, machine learning, and a thesis with Ericsson Research on anomaly detection and causal inference in 5G networks.",
  },
  {
    years: "2022 – today", kind: "work", title: "AI Lead & Computer Vision Engineer", slug: "arboair", org: "Arboair AB", place: "Linköping, Sweden",
    url: "https://arboair.com", image: "/images/work/drone-tree-crowns-closeup.jpg", imageAlt: "Close-up drone image of a forest road between a young plantation and mature trees, with each tree crown outlined and coloured by species",
    story: "Teaching machines to read forests from drone and aerial imagery: segmentation, geospatial ML, and leading the company's AI work.",
  },
];
