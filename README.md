# Project_1

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
-------------------
# Project_2
# QR Decomposition – Least Squares Classification (EMNIST Letters)

This repository contains an implementation of **QR decomposition–based least squares classification**.

The project focuses on solving a **multi-class classification problem** using **QR decomposition** and updating techniques for least-squares solutions.

### 1. Initial Model Training
- Load the **EMNIST Letters** dataset.  
- Select **200 training samples per class** and form the training matrix.  
- Solve the least squares classification problem:
  - Compute the **QR decomposition** of the training matrix (using Householder reflections).
  - Solve for the weight matrix using back-substitution.
- Evaluate model accuracy on a separate **test set**.

### 2. QR Updating
- Introduce **20 new samples per class** (simulating incremental data).
- Update the QR decomposition efficiently **without recomputing from scratch**, using QR updating techniques.
- Solve the updated least-squares system and re-evaluate performance.
- Compare accuracy **before and after updating**.


## Key Concepts
- **QR decomposition** with Householder reflections.
- **Incremental QR updating** for efficient retraining.
- **Least-squares classification** for multi-class problems.
- Performance comparison between initial model and updated model.

## Environment
This project can be run locally or in **Google Colab** using:
- `numpy` (linear algebra operations)
- `scipy` (optional for QR decomposition)
- `matplotlib` (visualization of results)
-----------------
# Project_3

# Clustering Analysis with NMF and K-Means

This repository presents a **clustering analysis on the 20 Newsgroups dataset**, focusing on five categories. The project explores **Non-Negative Matrix Factorization (NMF)**, a **custom K-Means implementation**, and **Scikit-learn’s K-Means**.

### Key Steps
- **Preprocessing:** stopword removal, punctuation/number filtering, tokenization, lowercasing  
- **Feature Extraction:** TF-IDF Term-Document matrix  
- **Clustering Methods:** NMF, custom K-Means, Scikit-learn K-Means  

### Results
- **Best Overall:** Scikit-learn K-Means (highest ARI & AMI)  
- **Moderate:** NMF (better intra-cluster cohesion)  
- **Weakest:** Custom K-Means (random centroid initialization issues)  
- **Challenge:** Low cluster separation in TF-IDF space

-------------
#  Project_4

This repository contains implementations and analyses for two main tasks:  
1. **Computing positions of Iranian cities in 2D space using ISOMAP**  
2. **Dimensionality reduction on the MNIST dataset using PCA and t-SNE**

---

##  Section 1: ISOMAP on Iranian Cities
- **Goal:** Map Iranian cities into 2D space while preserving geodesic distances.  
- **Approach:**
  - Manual ISOMAP implementation with KNN graph, Dijkstra’s algorithm, and MDS  
  - Scikit-learn’s `Isomap` for comparison  
- **Findings:**
  - Both implementations captured geographic structures (e.g., Tabriz, Urmia, Ardabil clustered together).  
  - Scikit-learn’s version produced smoother and more symmetric embeddings.  
  - Manual version showed acceptable results but with more distortion in scaling and positions.  

---

##  Section 2: Dimensionality Reduction on MNIST
- **Dataset:** 70,000 handwritten digit images (28×28 pixels).  
- **Steps:**
  - Flattened images into 784-dimensional vectors.  
  - Applied **PCA** (linear, variance preservation) and **t-SNE** (nonlinear, local structure preservation).  
  - Evaluated using **Trustworthiness metric** and explored **Perplexity effect** on t-SNE.  
- **Results:**
  - **Trustworthiness:** PCA = 0.5170, t-SNE = 0.6744 → t-SNE better preserved local structures.  
  - **Perplexity Analysis:** Increasing Perplexity lowered KL Divergence.  
    - Best result at **Perplexity = 50**, KL Divergence ≈ 1.70.  
- **Conclusion:**  
  - PCA is efficient but loses neighborhood details.  
  - t-SNE excels in preserving local structures, especially with optimized Perplexity.  
---------------
