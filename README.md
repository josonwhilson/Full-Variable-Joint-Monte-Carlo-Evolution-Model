# Alpha-Omega Dual-Factor Kinetic Simulation

This repository contains the source code and source data for the Monte Carlo simulations presented in our paper: 
**"[Transcending the Flory Limit: Decoupling Micro-Kinetic Heterogeneity from Statistical Laws in Polymer Functionalization]"**.

## Overview
This stochastic simulation models the reaction of functional groups on a polymer chain, incorporating Molecular Weight Distribution (MWD) derived from empirical GPC data, macro-triggering (Alpha), and micro-pairing (Omega) factors.

## Requirements
To run this code, please ensure you have Python 3.8+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```
## How to Run
1. Configure your parameters in the `.env` file.
2. Ensure your GPC data (e.g., `GPC_articaluse.xlsx`) is in the same directory.
3. Execute the main script:
```bash
python main.py
```
## Data Input
The model accepts empirical GPC data (Weight fraction dw/dlogM vs logM) and mathematically converts it to a number fraction distribution for accurate Monte Carlo sampling.
```
