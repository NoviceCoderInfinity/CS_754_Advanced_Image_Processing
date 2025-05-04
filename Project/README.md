# Course Project: Compressed Sensing Using Binary Matrices

This project explores compressed sensing using sparse binary matrices constructed from expander graphs, based on the paper:

> **"Compressed Sensing Using Binary Matrices of Nearly Optimal Dimensions"**  
> Mahsa Lotfi, Mathukumalli Vidyasagar

---

## 📘 Overview

Compressed Sensing (CS) enables the reconstruction of sparse signals from far fewer measurements than traditionally required. This project implements and analyzes binary measurement matrices based on expander graphs, which are memory-efficient and computationally faster alternatives to the baselines, random Gaussian matrices.

---

## 📁 Contents

- `Code/`: Python scripts and notebooks for simulation
- `Images/`: Mathematical derivations and LaTeX write-ups
- `Project_Proposal.pdf`: Initial deliverables promised at the beginning of the project
- `Report.pdf`: Final report in IEEE format containing detailed mathematical proofs, results and conclusion
- `README.md`: This file

---

## Methodology

The results in the paper were reproduced in the following fashion. For the purpose of measurement matrices:

- `Array Code` (LDPC - Low Density Parity Checker)
- `Euler Matrix`
- `Guassian Matrix`

The reconstruction was evaluated by compressing the original signal and then by using the following metrics

- `Timing Analysis`
- `Phase Transition Analysis`
- `Structural Similarity Index Metric (SSIM)`

---

## 🧪 Run the Code

The code has been written in MATLAB and optimized by using CVX package. Firstly install any MATLAB versions (R2007b or later). Next, install the cvx package, and follow the steps listed [here](https://www.cse.iitb.ac.in/~cs709/notes/code/cvx/doc/install.html)

```
Code/
|---ArrayCode_vs_Gaussian_timing_analysis.m
|---Euler_timing_analysis.m
|---Phase_transition.m
|---SSIM_vs_k.m
```

---

## Results

![Phase Transition Graph for Array Code](Images/Phase%20Transition_Array%20code.png)

![Phase Transition Graph for Guassian](Images/Phase%20Transition_Gaussian.png)

![SSIM for Array Code](Images/SSIM%20vs%20k_ArrayCode.png)

![Euler for Array Code](Images/SSIM%20vs%20k_Euler.png)

![SSIM for Guassian](Images/SSIM%20vs%20k_Gaussian.png)
