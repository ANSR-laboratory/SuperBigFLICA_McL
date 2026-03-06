# SuperBigFLICA (SBF)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![medRxiv](https://img.shields.io/badge/medRxiv-2025.12.11.25341830-blue)](https://www.medrxiv.org/content/10.64898/2025.12.11.25341830v1)

> **Semi-supervised multimodal neuroimaging data fusion** that simultaneously learns interpretable spatial brain components and predicts clinical outcomes — enabling both discovery and prediction from multi-modal brain imaging data.

<p align="center">
  <img src="figures/overview.png" width="85%" alt="SuperBigFLICA overview">
</p>

## Citation

SuperBigFLICA was originally proposed in:

```bibtex
@article{gong2023supervised,
  title={Supervised Phenotype Discovery From Multimodal Brain Imaging},
  author={Gong, Weikang and Bai, Shuang and Zheng, Yong-Qiang and Smith, Stephen M. and Beckmann, Christian F.},
  journal={IEEE Transactions on Medical Imaging},
  volume={42},
  number={3},
  pages={834--849},
  year={2023},
  doi={10.1109/TMI.2022.3218720}
}
```

If you use this codebase in your research, please also cite:

```bibtex
@article{cheng2025sbf,
  title={Investigating the Amyloid-Tau-Neurodegeneration Framework in Alzheimer's Disease Using Semi-Supervised Multimodal Imaging Data Fusion},
  author={Cheng, You and Medina, Adri{\'a}n and Korponay, Cole and Beckmann, Christian F. and Harper, David and Nickerson, Lisa and {Alzheimer's Disease Neuroimaging Initiative}},
  journal={medRxiv},
  year={2025},
  doi={10.64898/2025.12.11.25341830},
  url={https://www.medrxiv.org/content/10.64898/2025.12.11.25341830v1}
}
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ANSR-laboratory/SuperBigFLICA_McL.git
cd SuperBigFLICA_McL

# Install dependencies
pip install -r requirements.txt         # Linux (Python ≥ 3.11)
pip install -r requirements.macos.txt   # macOS (Python ≥ 3.10)

# Configure paths and parameters in Scripts/SBF.py, then run:
python Scripts/SBF.py
```

## Project Structure

```
SuperBigFLICA_McL/
├── Scripts/                # Main scripts (SBF.py, sbf_utils.py, RUN_SBF_sbatch.sh)
├── Data/                   # Input data folder
├── figures/                # Overview figures
├── Slides/                 # Presentation materials
├── Visualizations/         # Output visualization utilities
├── model_cards/            # Documentation for trained models
├── requirements.txt        # Linux dependencies
└── requirements.macos.txt  # macOS dependencies
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
