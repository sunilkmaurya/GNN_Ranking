# Code repository for GNN_Ranking

This repository contains the code for our paper "Graph Neural Networks for Fast Node Ranking Approximation" accepted at ACM Transactions on Knowledge Discovery from Data(TKDD). This paper is an extension of our previous paper "[Fast Approximations of Betweenness Centrality using Graph Neural Networks](https://dl.acm.org/doi/10.1145/3357384.3358080)", which is accepted as a short research paper in CIKM 2019.
In this paper, we extend our proposed framework to approximate both betweenness and closeness centrality for the directed graphs. Also, we demonstrate the usefulness of our framework in case of the dynamic graphs.

Code is written in Python and the proposed model is implemented using Pytorch.

**Main package Requirements**  
Pytorch, NetworKit, NetworkX and SciPy.  
Use of conda environment is recommended, but not necessary.
Experiments in paper used PyTorch (0.4.1) and Python (3.7).

**Running the model code**:  
There are two variants of our model: Betweenness Centrality variant and Closeness Centrality variant. These can be run via two scripts `betweenness.py` and `closeness.py`. To test the model, we have provided some synthetic graph datasets. These datasets are based on three different directed graph types: Scale-free (SF), Erdos-Renyi (ER) and Gaussian Random Partition graphs (GRP). 

To run betweenness variant (on scale-free datasets) :
```
python betweenness.py --g SF
```

For closeness variant (on scale-free datasets):
```
python closeness.py --g SF
```

"SF" can be replaced by "ER" or "GRP".

**Dataset Details**:
Graphs are created using script `./datasets/generate_graph.py`, which generates graphs with number of nodes varying from 5,000 to 10,000 (randomly chosen). Graph generation parameters are also randomly selected. After generation of graphs, betweenness and closeness centrality are calculated and stored in pickle file in folder `./datasets/graphs/`.

Training and test splits are created using `./datasets/create_dataset.py`. 40 graphs are used for training and 10 graphs for testing. Training samples are created by permuting node sequence while generating adjacency matrices from the graphs (training). Training and testing samples are stored as `training.pickle` and `test.pickle` in folders named corresponding to graph-type at `./datasets/data_splits/`.

**Addition**:  
I have added a Jupyter notebook to show some interesting observations. We simplify our model to single layer and train it rank nodes based on degree centrality. We see that the trained model can easily rank the nodes similar to degree centrality in new different types of graphs without being provided any explicit information. In current literature, it has already been discussed that the Graph Neural Networks (GNN) may learn to distinguish the node degree. Hence, these observations are not discussed in the paper. But we show it experimentally here.

 In short, the model is able to rank nodes in the graph corresponding to **degree centrality**, **betweenness centrality** and **closeness centrality**.

 
**Note (PyTorch 1.0 or higher)**:  
This code was written and tested on PyTorch (0.4.1), so it has some incompatibilities with newer versions. With PyTorch versions (1.0 or higher), this code may give inconsistence performance. This is because of some of the changes in newer versions cause problems with this code. One reason is dropout not acting as intended in the code (See [https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3](https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3)).
For example, changing the dropout code (at two locations) in MLP layer in `layer.py`,
```
score_temp = F.dropout(score_temp,self.dropout)
```
to (for newer PyTorch versions)
```
score_temp = F.dropout(score_temp,self.dropout,self.training)
```
improves the performance similar to original results. In addition to this fix, I found results varying a bit (between older and newer versions) even with same random seed. I will look into it and provide a patch for newer PyTorch versions.

**Note about package requirements**

In case there are conflicts while installing the packages, please check the following [issue](https://github.com/sunilkmaurya/GNN_Ranking/issues/2) for matched versions of the packages. Thanks @natema for the information. 

