// Longer stories behind each Journey stop, shown on /journey/<slug>/.
// Course lists are names only: no grades, transcripts or certificates are published.

export type Fact = { label: string; value: string };
export type Block = { heading: string; paragraphs?: string[]; bullets?: string[] };
export type CourseGroup = { title: string; note?: string; items: string[] };
export type LinkItem = { label: string; href: string; note?: string };
export type Photo = { src: string; alt: string };

export type StopDetail = {
  intro: string;
  facts: Fact[];
  explainer?: { title: string; paragraphs: string[] };
  blocks: Block[];
  courses?: { title: string; intro?: string; groups: CourseGroup[] };
  links?: { title: string; items: LinkItem[] };
  photos?: Photo[];
  closingPhoto?: Photo;
  story?: "acquisition";
};

export const details: Record<string, StopDetail> = {
  "bsc-petroleum-exploration": {
    intro:
      "Four years in Ahvaz learning how people find oil and gas before anyone drills a well. This is where I first met geophysics, and where waves travelling through rock became something I wanted to understand properly.",
    facts: [
      { label: "Degree", value: "BSc, Petroleum Exploration Engineering" },
      { label: "Length", value: "4 years, 8 semesters" },
      { label: "Where", value: "Petroleum Engineering Faculty, Ahvaz" },
      { label: "Internships", value: "2 (summer 2003)" },
    ],
    explainer: {
      title: "What is petroleum exploration engineering?",
      paragraphs: [
        "Oil and gas sit kilometres underground, trapped in porous rock. Drilling a well costs a lot, so before anyone drills you want to be fairly sure something is down there.",
        "The degree mixes three things. Geology tells you which rocks can hold oil and how they got folded and buried. Geophysics lets you \"see\" underground without digging, mostly by sending waves into the ground and listening to the echoes. And petroleum engineering covers what happens after: drilling, reservoirs and getting the oil out.",
      ],
    },
    blocks: [
      {
        heading: "What I did there",
        bullets: [
          "Took the full mix of geology, geophysics and petroleum engineering courses (listed below).",
          "Two internships in 2003: one with the exploration directorate of the national oil company, on seismic acquisition and interpretation, and one on well-site logging.",
          "Gave a talk on 4D seismic, which is repeating a seismic survey over time to watch a reservoir change.",
          "Was a student representative and ran a student magazine about petroleum exploration.",
        ],
      },
      {
        heading: "What stayed with me",
        paragraphs: [
          "The geophysics courses were the ones that clicked. The idea that you could turn recorded vibrations into a picture of rock layers is basically the same idea I still work with, just with cameras instead of geophones.",
        ],
      },
    ],
    courses: {
      title: "Main courses",
      intro: "The core subjects, grouped by area. General and required university courses are left out.",
      groups: [
        { title: "Geophysics and petrophysics", items: ["Geophysics I", "Geophysics II", "Geophysics lab", "Well logging", "Petrophysics", "Rock mechanics"] },
        {
          title: "Geology",
          items: [
            "General geology", "Mineralogy", "Petrology", "Paleontology", "Micropaleontology", "Sedimentology", "Stratigraphy",
            "Sequence stratigraphy", "Structural geology", "Tectonics", "Subsurface geology", "Geological mapping", "Petroleum geology",
            "Organic geochemistry", "Engineering geology", "Geology of Iran", "Geology field work",
          ],
        },
        {
          title: "Petroleum engineering",
          items: [
            "Fundamentals of petroleum engineering", "Petroleum reservoir engineering", "Estimation and evaluation of petroleum reservoirs",
            "Enhanced oil recovery", "Principles of drilling", "Drilling mud and well cementation", "Fluid mechanics", "Technical and economic analysis",
          ],
        },
        {
          title: "Maths, physics and computing",
          items: ["General mathematics I and II", "Differential equations", "Engineering mathematics", "Engineering statistics and probability", "Physics I and II", "Computer programming"],
        },
        { title: "Projects", items: ["BSc project", "Apprenticeship I and II"] },
      ],
    },
  },

  nisoc: {
    story: "acquisition",
    intro:
      "Fourteen years in the oil fields of south-west Iran, making sure seismic surveys recorded clean, trustworthy data. I started as a junior QC geophysicist and left as a senior supervisor, and most of my habits around data come from here.",
    facts: [
      { label: "Role", value: "Seismic Data Acquisition QC Supervisor" },
      { label: "Years", value: "2005 – 2019, junior to senior" },
      { label: "Where", value: "Oil fields in Khuzestan, Iran" },
      { label: "Surveys", value: "2D and 3D land, in more than five oil fields" },
    ],
    explainer: {
      title: "What is seismic acquisition?",
      paragraphs: [
        "Think of it as an ultrasound scan of the earth. A source sends a pulse of energy into the ground: either big vibroseis trucks that shake the ground with a controlled sweep, or small dynamite charges in shallow holes.",
        "The waves travel down, and every time they hit a boundary between rock layers, part of the energy bounces back up. Thousands of small sensors called geophones, laid out in lines on the surface, record those echoes. A recording truck collects everything.",
        "Later, processing turns millions of these recordings into an image of the layers underground, and geologists use that image to decide where to drill. A 2D survey records along single lines and gives a cross-section. A 3D survey covers a whole area with a dense grid and gives a full volume.",
      ],
    },
    blocks: [
      {
        heading: "What quality control means here",
        paragraphs: [
          "A survey is only worth something if the data is clean and every recording is where it's supposed to be. If a problem slips through, it shows up months later as a blurry or wrong image, and by then the crew is gone.",
          "So QC is checking, every day: is the source working properly, are the sensors all alive and wired the right way, are the positions right, is noise under control, and are the field procedures and safety rules followed. When something is wrong, you decide on the spot whether to fix it or record again.",
        ],
      },
      {
        heading: "What I did",
        bullets: [
          "Quality control of 2D and 3D land seismic surveys in deserts, sand dunes and folded mountains: recording quality, survey geometry and field procedures.",
          "Supervised acquisition in more than five oil fields, with both dynamite and vibroseis sources.",
          "Reviewed survey designs and field data before they went to processing.",
          "Wrote technical reports on survey-design scouting, receiver-array design, recording in a busy industrial city area, and safe distances for blasting near buildings.",
          "Mentored new and junior geophysicists, and advised an MSc student on her thesis.",
          "Was recognised as one of the company's best seismic acquisition supervisors in 2015.",
        ],
      },
      {
        heading: "What carried over into computer vision",
        paragraphs: [
          "More than I expected. Waves, spectra, filtering and noise are the same ideas whether the signal is a seismic trace or an image. Checking data before trusting it is most of the work in training data too. And survey layouts, coordinates and terrain turned into orthomosaics, map projections and GIS.",
        ],
      },
    ],
    courses: {
      title: "Training courses during the job",
      intro: "Professional courses I completed while working at NISOC.",
      groups: [
        {
          title: "Survey design",
          items: [
            "Advanced 2D–3D design for land seismic operations and parameter design (Prospectiuni, Bucharest)",
            "Fundamentals of 3D seismic survey design (3DSymSam)",
            "3D seismic survey design with OMNI Workshop 5.0 (Kish, 2006)",
          ],
        },
        {
          title: "Processing and interpretation",
          items: [
            "Seismic data processing (Research Institute of Petroleum Industry, Tehran, 2008)",
            "Advanced seismic data processing",
            "Advanced seismic data interpretation (Prospectiuni)",
            "Vertical seismic profiling: tools, design, processing and interpretation (Petropars, 2010)",
          ],
        },
      ],
    },
    links: {
      title: "Publications from the field years (in Persian)",
      items: [
        {
          label: "Applying the asymmetrical extended-patch method to improve data quality in a 3D seismic acquisition project",
          href: "",
          note: "Kimassi, Mansuri, Pirmoradiyan. Journal of Exploration & Production Oil & Gas, No. 143, 2017.",
        },
        {
          label: "Optimising a 3D seismic survey design to acquire high-quality data",
          href: "",
          note: "Kimassi, Paryab, Pirmoradiyan, Puladzadeh. NISOC Journal of Knowledge Management, 2016.",
        },
        {
          label: "The origin of noise events in surface seismic data and how to reduce them",
          href: "",
          note: "Pirmoradiyan, Tayyebi. NISOC Journal of Knowledge Management, 2016.",
        },
      ],
    },
    photos: [
      { src: "/images/field/field-survey-with-maps.jpg", alt: "Scouting a survey area with the line maps" },
      { src: "/images/field/folded-mountain-geology.jpg", alt: "Folded mountain geology, the target of the surveys" },
      { src: "/images/field/field-camp.jpg", alt: "The field camp" },
    ],
    closingPhoto: { src: "/images/field/qc-station-in-recording-truck.jpg", alt: "At my QC desk in the recording truck" },
  },

  "msc-geophysics": {
    intro:
      "A master's degree in seismology, done in the evenings while I kept working in the field. It gave me the theory behind the data I was checking every day, and my thesis was about turning seismic recordings into rock properties.",
    facts: [
      { label: "Degree", value: "MSc, Geophysics (Seismology)" },
      { label: "Length", value: "2009 – 2012, evening programme" },
      { label: "Where", value: "Institute of Geophysics, University of Tehran" },
      { label: "Graduated", value: "January 2012" },
    ],
    explainer: {
      title: "What is seismic inversion?",
      paragraphs: [
        "A normal seismic image shows where the boundaries between rock layers are, because that's where the waves bounce back. But it doesn't directly tell you what the rock is like between those boundaries.",
        "Inversion works backwards. It asks: what layers of rock, with what properties, would produce exactly these recordings? The main property is acoustic impedance, which is roughly how \"hard\" the rock is for a sound wave (density times velocity).",
        "Prestack inversion goes a step further. It uses how the reflections change with the angle of the wave, and that lets you separate several properties. Those properties give hints about porosity and about what fluid is in the rock, oil, gas or water.",
      ],
    },
    blocks: [
      {
        heading: "My thesis",
        paragraphs: [
          "Petroleum reservoir characterization using seismic data analysis. I ran prestack inversion on seismic data from an Iranian oil field and then interpreted the resulting attributes to describe the reservoir.",
          "The hard part wasn't running the software. It was understanding what the numbers meant and when to not trust them, which is a lesson I keep relearning in machine learning.",
        ],
      },
      {
        heading: "Studying and working at the same time",
        paragraphs: [
          "The programme ran in the evenings, so field work continued during the day. It was tiring, but it meant everything I learned in class I could check against real data almost the next week.",
        ],
      },
    ],
    courses: {
      title: "Courses",
      groups: [
        { title: "Seismology and waves", items: ["Seismology I", "Seismology II", "Theory of elastic waves", "Engineering seismology"] },
        { title: "Reflection seismic", items: ["Reflection seismic operations", "Reflection seismic data processing", "Interpretation of reflection seismic data"] },
        { title: "Other geophysics and methods", items: ["Digital filters", "Numerical analysis", "Geoelectrics", "Well logging"] },
        { title: "Research", items: ["Seminar", "MSc thesis"] },
      ],
    },
  },

  "msc-statistics-ml": {
    intro:
      "In 2019, after fourteen years in industry, I moved to Sweden to study statistics and machine learning full-time. Two intense years: lots of theory, lots of labs, and a thesis with Ericsson Research at the end.",
    facts: [
      { label: "Degree", value: "MSc, Statistics and Machine Learning" },
      { label: "Length", value: "2 years full-time (Aug 2019 – Jun 2021)" },
      { label: "Credits", value: "120 ECTS, 16 courses plus a 30-credit thesis" },
      { label: "Where", value: "Division of Statistics and Machine Learning, Linköping University" },
    ],
    blocks: [
      {
        heading: "What it was like",
        paragraphs: [
          "Going back to university in a new country, in a new language, after a long career was a big step. The programme is very hands-on: almost every course had computer labs and reports in R or Python, so you learn by building things, not only by reading.",
          "Coming from geophysics helped more than I thought. Signals, noise, inversion and uncertainty were already familiar ideas. Machine learning gave them new names and much better tools.",
        ],
      },
      {
        heading: "My thesis with Ericsson Research",
        paragraphs: [
          "Anomaly detection and causal inference in 5G mobile networks. I did it in spring 2021 with Ericsson AI Research, supervised by Jose M. Peña at Linköping University and Caner Kilinc at Ericsson.",
          "The problem: 5G promises very low latency, the delay between sending a message and getting it through. When the delay suddenly gets high, engineers want two things. They want to catch it, and they want to know why it happened, so they can fix the right thing.",
          "I worked with measurements from an Ericsson test network. First I trained a random forest classifier to flag the high-latency moments. Then, for the \"why\" part, I used causal Bayesian networks. The PC algorithm learns a graph of which measurements influence which, and from that graph you can estimate how much each one actually causes the latency to change, not just how much it moves together with it.",
          "The classifier could flag a good share of the high-latency cases, and the causal analysis pointed to a small set of network measurements as the main drivers of the delay. The biggest lesson for me was the gap between \"correlated with\" and \"causes\". It's easy to say and surprisingly hard to get right with real data.",
          "The data belongs to Ericsson and is confidential, so the published report renames the variables and only describes the data in general terms.",
        ],
      },
    ],
    links: {
      title: "Thesis report",
      items: [
        {
          label: "Anomaly detection and causal inference in 5G mobile networks (PDF)",
          href: "/docs/2021_Pirmoradiyan_MSc_thesis_anomaly_detection_causal_inference_5G.pdf",
          note: "Master's thesis, Linköping University, 2021 (LIU-IDA/STAT-A--21/040--SE). The public university version.",
        },
      ],
    },
    courses: {
      title: "Courses, semester by semester",
      groups: [
        { title: "Autumn 2019", items: ["Statistical methods", "Advanced programming in R", "Machine learning", "Multivariate statistical methods", "Advanced academic studies"] },
        { title: "Spring 2020", items: ["Computational statistics", "Advanced data mining", "Introduction to Python", "Bayesian learning", "Big data analytics", "Deep learning"] },
        { title: "Autumn 2020", items: ["Probability theory", "Time series and sequence learning", "Advanced machine learning", "Database technology", "Text mining"] },
        { title: "Spring 2021", items: ["Master's thesis in statistics (30 credits), with Ericsson Research"] },
      ],
    },
  },

  arboair: {
    intro:
      "Since 2022 I've worked at Arboair, a Swedish company that analyses forests from drone and aerial images. I joined as a computer-vision engineer and now lead the company's AI work.",
    facts: [
      { label: "Role", value: "AI Lead & Computer Vision Engineer" },
      { label: "Since", value: "February 2022" },
      { label: "Where", value: "Linköping, Sweden" },
      { label: "Focus", value: "Segmentation, geospatial ML, leading AI work" },
    ],
    explainer: {
      title: "What does forest analytics from above mean?",
      paragraphs: [
        "Forest owners need to know what's in their forest: which tree species grow where, how big the trees are, and where there's damage from storms or insects. Walking through every stand is slow and expensive.",
        "Drone and aerial images can cover large areas quickly. Computer vision turns those images into maps, for example outlining each tree crown and saying which species it is, so the result can go straight into a map or a forest plan.",
      ],
    },
    blocks: [
      {
        heading: "What I do, in short",
        bullets: [
          "Designed and lead the company's computer-vision platform, from training data to models running in production.",
          "Train deep-learning segmentation models for tree species and forest damage.",
          "Work on the geospatial side that turns pixel predictions into maps.",
          "Lead the AI work across the team.",
        ],
      },
    ],
    links: {
      title: "More about this role",
      items: [{ label: "Read the full story on the Work page", href: "/work/" }],
    },
  },
};
