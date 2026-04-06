# Training-Free Refinement of Flow Matching with Divergence-based Sampling

<div align="center">

[Yeonwoo Cha](https://yeonwoo378.github.io) &nbsp;·&nbsp;
[Jaehoon Yoo](https://sites.google.com/view/jaehoon-yoo/) &nbsp;·&nbsp;
[Semin Kim](https://seminkim.github.io) &nbsp;·&nbsp;
[Yunseo Park](https://github.com/pys0622) &nbsp;·&nbsp;
[Jinhyeon Kwon](https://github.com/jinhyeonkwon) &nbsp;·&nbsp;
[Seunghoon Hong](https://maga33.github.io)

**KAIST**

[![Paper](https://img.shields.io/badge/arXiv-2604.?????-b31b1b.svg)](https://arxiv.org/abs/?)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://yeonwoo378.github.io/official_fds)

</div>

<br>

![Overview](assets/fds_main_png.png)

---

## Overview

We propose **Flow Divergence Sampler (FDS)**, a training-free inference-time enhancement for diffusion and flow matching models. FDS improves the overall quality of generated samples by selecting lower-divergence trajectories at each ODE timestep, using a Hutchinson divergence estimator — with no additional training required.

> **FDS is backbone-agnostic** — it can be applied to any diffusion or flow matching model with minimal integration effort. This repository provides an implementation built around **JiT** (Just-image Transformer), a class-conditional pixel-space image generation model trained on ImageNet-256, which serves as the backbone for our main experiments.

---

## Installation

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended).

```bash
# 1. Clone the repository
git clone https://github.com/yeonwoo378/flow-divergence-sampler.git
cd flow-divergence-sampler

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Inference

Try FDS with running our demo now!

```bash
jupyter notebook run_FDS.ipynb
```



---

## Pretrained Checkpoints

Download the pretrained JiT checkpoints and place them in the `checkpoints/` directory as follows:

```
flow-divergence-sampler/
└── checkpoints/
    ├── jit-b-16/
    │   └── checkpoint-last.pth      # JiT-B/16  (~131M params)
    ├── jit-l-16/
    │   └── checkpoint-last.pth      # JiT-L/16  (~131M params)
    └── jit-h-16/
        └── checkpoint-last.pth      # JiT-H/16  (best quality)
```




---

## Acknowledgements

This repository builds upon the following excellent open-source projects:

- [JiT](https://github.com/LTH14/JiT)
- [EDM](https://github.com/NVlabs/edm)
- [FFJORD](https://github.com/rtqichen/ffjord)

---

## Citation

If you find this work helpful, please cite:

```bibtex
@misc{cha2026fds,
  title   = {Training-Free Refinement of Flow Matching with Divergence-based Sampling},
  author  = {Cha, Yeonwoo and Yoo, Jaehoon and Kim, Semin and Park, Yunseo and Kwon, Jinhyeon and Hong, Seunghoon},
  journal = {arXiv preprint arXiv:2604.?????},
  year    = {2026}
}
```

---

## Contact

For any inquiries, please contact **Yeonwoo Cha** at `ckdusdn03@kaist.ac.kr`.
