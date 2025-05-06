# Cold Pool Detection and Tracking

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10115957.svg)](https://doi.org/10.5281/zenodo.10115957)

This repository contains the code implementation for the detection and tracking of cold pools as described in the peer-reviewed article:

**Hoeller, J., Fiévet, R., & Haerter, J. O. (2024). Detecting cold pool family trees in convection resolving simulations. Journal of Advances in Modeling Earth Systems, 16, e2023MS003682.**  
📄 [Read the full article here](https://doi.org/10.1029/2023MS003682)

---

## 🧊 Overview

Cold pools are crucial mesoscale phenomena influencing convection and boundary layer dynamics. This repository provides an efficient, physically interpretable Python-based algorithm for detecting and tracking cold pools in large-eddy simulation (LES) output.

The code was developed in the context of the paper above and is structured to replicate all analysis and figures using the provided data.

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Shakiro7/coldPool-detection-and-tracking.git
cd coldPool-detection-and-tracking
```

### 2. Create a Virtual Environment *(optional but recommended)*

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Example Dataset

The dataset used in the publication is hosted on Zenodo:

📦 [Zenodo Dataset (10.5281/zenodo.10115957)](https://zenodo.org/records/10115957)

Download the simulation files (`*.nc`) and place them in a suitable data directory. Update the paths in the `main.py` accordingly.

### 5. Run the Algorithm

```bash
python main.py
```

The results, including plots and tracking outputs, can be saved in the specified output directory.

---

## 📊 Reproducing Paper Results

All analysis and figures from the paper can be reproduced using this code and the Zenodo dataset. For detailed instructions, refer to the comments and documentation within `main.py` as well as the paper.

---

## 📖 Citation

If you use this code or data in your work, please cite the following:

Paper
```bibtex
@article{https://doi.org/10.1029/2023MS003682,
author = {Hoeller, Jannik and Fiévet, Romain and Haerter, Jan O.},
title = {Detecting Cold Pool Family Trees in Convection Resolving Simulations},
journal = {Journal of Advances in Modeling Earth Systems},
volume = {16},
number = {1},
pages = {e2023MS003682},
keywords = {cold pools, detection, tracking, cloud resolving simulation, convective organization},
doi = {https://doi.org/10.1029/2023MS003682},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023MS003682},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023MS003682},
note = {e2023MS003682 2023MS003682},
year = {2024}
}
```

Software and dataset
```bibtex
@software{hoeller_2023_10115957,
  author       = {Hoeller, Jannik},
  title        = {CoolDeTA: Detection and Tracking of Convective
                   Cold Pools and Their Causal Chains in Cloud-
                   Resolving Simulation Data
                  },
  month        = nov,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.10115957},
  url          = {https://doi.org/10.5281/zenodo.10115957},
}
```

---

## 📬 Contact

For questions, suggestions, or collaborations, feel free to reach out.
