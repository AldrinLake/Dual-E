# Dual-Estimator: Decoupling Global and Local Semantic Shift for Drift Compensation in Class-Incremental Learning

> **Notice:** The codebase is currently being reorganized and will be released after the paper is online.

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](http://cvpr.thecvf.com/)  [![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-orange)](https://pytorch.org/)

> ##### **Accepted by CVPR 2026**
>
> This repository provides the official implementation for Dual-E, a drift compensation method for Exemplar-Free Class-Incremental Learning (EFCIL).

## Overview

In EFCIL, it is common to retain intermediate representations (e.g., class prototypes) instead of raw samples. As the backbone evolves, these representations drift. Most drift compensation methods assume uniform semantic distributions and uniform semantic shifts, which is unrealistic under random class streams.

**Dual-E** addresses this by decoupling *local* and *global* semantic shifts with two complementary estimators:

- **MoE-Estimator (MoE-E)**: uses multiple experts to model local shift patterns, reducing bias from intra-task non-uniform semantic distributions.
- **Low-Rank Estimator (LoR-E)**: models global shift patterns with low-rank structure, stabilizing compensation for classes with large semantic gaps.

![1772515390437](images/README/1772515390437.png)

##### Dual-E relies on analytical updates, is computationally efficient, and can be plugged into existing EFCIL pipelines.

---

## Paper and Citation

This work has been **accepted by CVPR 2026**. The official BibTeX entry will be released with the public version of the paper.

---

## Acknowledgement

This project is mainly based on [PyCIL](https://github.com/LAMDA-CL/PyCIL).
