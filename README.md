# PageRank Implementation

This repository contains implementations of the **PageRank algorithm**.

The goal of this project is to compute and compare the **PageRank scores** for nodes in the **Stanford-Berkeley Web Graph dataset**. The implementation is done in three stages:

1. **Custom PageRank (Power Iteration)**  
   - Implemented from scratch using the Power Iteration method.  
   - Produces PageRank scores and reports the top-ranked pages by ID.  

2. **Eigenvector-based PageRank**  
   - Uses `numpy.linalg.eig` to compute eigenvectors of the transition matrix.  
   - Compares the results and runtime with the Power Iteration method.  

3. **NetworkX Comparison**  
   - Uses built-in functions from `networkx` to compute PageRank.  
   - Compares and interprets differences with the custom implementation.  

Additionally, the effect of **varying the damping factor α** is explored and analyzed to study its influence on the final ranking.

## Dataset
We use the [Stanford-Berkeley Web Graph dataset](https://www.kaggle.com/datasets/wolfram77/graphs-snap-web/data), but only process the **first 1000×1000 submatrix** (`Data[:1000, :1000]`) due to its large size.  

## Environment
The code can be executed on **Google Colab** or locally with Python. If additional compute resources are required, Colab is recommended.  

## Key Learnings
- Understanding of **Markov Chains** and **Random Walks** as the foundation of PageRank.  
- Hands-on experience with iterative methods (Power Iteration) vs. direct eigenvalue computation.  
- Insights into how damping factor tuning changes the ranking distribution. 
