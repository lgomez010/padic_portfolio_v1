# p-Adic Portfolio Optimization

## Research Objective
This repository contains a modular Python package for modeling financial market dynamics by embedding stock correlation data into a $p$-adic space, specifically $\mathbb{Z}_2$. By deriving a subdominant ultrametric from a Minimum Spanning Tree (MST) of asset returns, this model extracts hierarchical market structures to generate optimized portfolio weights via Hierarchical Risk Parity (HRP).

## Project Architecture
The project is structured as a standard Python package (`padic_portfolio`) to ensure modularity and mathematical rigor:

* **`padic_portfolio/topology/metric.py`**: The core mathematical engine. Contains functions to transform empirical correlation matrices into distances, compute the subdominant ultrametric, and execute the isometric embedding into $\mathbb{Z}_2$.
* **`padic_portfolio/allocation/hrp.py`**: The portfolio construction module. Utilizes the hierarchical topological structures derived in `metric.py` to calculate optimized asset weights using Hierarchical Risk Parity.
* **`notebooks/`**: Contains Jupyter notebooks for exploratory data analysis, pipeline testing, and rendering the final backtest wealth curves.
* **`tests/`**: Unit testing suite utilizing `pytest` to ensure the mathematical stability of the ultrametric transformations and weight allocations.

## Environment Setup
Ensure you have a standard Python 3.x environment. Install the necessary scientific dependencies via:

```bash
pip install -r requirements.txt