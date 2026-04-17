[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/topometry/badge/?version=latest)](https://topometry.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/personalized-badge/topometry?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/topometry)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?style=social&label=Follow%20%40davisidarta)](https://twitter.com/davisidarta)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

# About TopoMetry

> **Fork notice:** This is a heavily modified fork of
> [original-repo](https://github.com/davisidarta/topometry) by David S Oliveira,
> It has diverged significantly and may or may not be maintained independently.

**TopoMetry** is a geometry-aware Python toolkit for exploring high-dimensional data via diffusion/Laplacian operators. It learns **neighborhood graphs → Laplace–Beltrami–type operators → spectral scaffolds → refined graphs** and then finds clusters and builds low-dimensional layouts for analysis and visualization.

- **scikit-learn–style transformers** with a high-level `TopOGraph` orchestrator
- **Fixed-time & multiscale spectral scaffolds**
- **Operator-native metrics** to quantify geometry preservation and **Riemannian diagnostics** to evaluate distortion in visualizations
- Designed for **large, diverse datasets**

For background, see the preprint: https://doi.org/10.1101/2022.03.14.484134

## Geometry-first rationale (short)

We approximate the **Laplace–Beltrami operator (LBO)** by learning well-weighted similarity graphs and their Laplacian/diffusion operators. The **eigenfunctions** of these operators form an orthonormal basis—the **spectral scaffold**—that captures the dataset’s intrinsic geometry across scales. This view connects to **Diffusion Maps**, **Laplacian Eigenmaps**, and related kernel eigenmaps, and enables downstream tasks such as clustering and graph-layout optimization with geometry preserved.

## When to use TopoMetry

Use TopoMetry when you want:

- Geometry-faithful representations beyond variance maximization (e.g., PCA)
- Robust low-dimensional views and clustering from operator-grounded features
- Quantitative **operator-native** metrics to compare methods and parameter choices
- Reproducible, **non-destructive** pipelines

Empirically, TopoMetry often outperforms PCA-based pipelines and stand-alone layouts. Still, **let the data decide**—TopoMetry includes metrics and reports to support evidence-based choices.

### When not to use TopoMetry

- **Very small sample sizes** where the manifold hypothesis is weak
- Workflows needing **streaming/online** updates or **inverse transforms** (embedding new points without recomputing operators is not currently supported). If that’s critical, consider UMAP or parametric/autoencoder approaches—and you can still use TopoMetry to **audit geometry** or **estimate intrinsic dimensionality** to guide model design.

## Installation

Prior to installing TopoMetry, make sure you have [cmake](https://cmake.org/), [scikit-build](https://scikit-build.readthedocs.io/en/latest/) and [setuptools](https://setuptools.readthedocs.io/en/latest/) available in your system. If using Linux:
```
sudo apt-get install cmake
pip install scikit-build setuptools
```

Then you can install TopoMetry from PyPI:

```
pip install topometry
```


## Tutorials and documentation

Check TopoMetry's [documentation](https://topometry.readthedocs.io/en/latest/) for tutorials, guided analyses and other documentation.



## Minimal example

```python
import numpy as np
import topo as tp

# Generate sample data (e.g., points on a Swiss roll)
from sklearn.datasets import make_swiss_roll
X, color = make_swiss_roll(n_samples=2000, noise=0.5, random_state=42)

# Learn topological metrics and spectral scaffold
tg = tp.TopOGraph()
tg.fit(X)

# Build a refined topological graph
tgraph = tg.transform(X)

# Optimize a 2-D layout
tg.project_graph_layouts()
```

## Changelog

**v2.0.0** — Core-only release
- Removed single-cell / scanpy / AnnData wrappers (now a standalone geometry toolkit)
- Core API unchanged: `TopOGraph`, spectral scaffolds, graph operators, layouts, metrics, plotting

#### Citation

---

```
@article {Oliveira2022.03.14.484134,
	author = {Oliveira, David S and Domingos, Ana I. and Velloso, Licio A},
	title = {TopoMetry systematically learns and evaluates the latent geometry of single-cell data},
	elocation-id = {2022.03.14.484134},
	year = {2025},
	doi = {10.1101/2022.03.14.484134},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/10/15/2022.03.14.484134},
	eprint = {https://www.biorxiv.org/content/early/2025/10/15/2022.03.14.484134.full.pdf},
	journal = {bioRxiv}
}
```
