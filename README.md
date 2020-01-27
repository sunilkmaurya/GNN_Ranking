# Code repository for GNN_Ranking

This repository contains code for our paper "Graph Neural Networks for Fast Node Ranking Approximation".

Code is written in python and the proposed model is implemented using Pytorch.

**Main package Requirements**: Pytorch, networkit, networkx and scipy.
Use of conda environment is recommended but not necessary.
Experiments in paper used PyTorch(0.4.1) and Python(3.7).

**Running the code**:

There are two variants of our model: One for betweenness and other for closeness. 


There are three different synthetic graph datasets: Scalefree (SF), Erdos-Renyi (ER) and Gaussian Random Partition graphs (GRP). 

To run betweenness variant :
```python betweenness.py --g SF```

For closeness variant :
```python closeness.py --g SF```

"SF" can be replaced by "ER" or "GRP".

I will updating this readme and put more details very soon(in one or two days).