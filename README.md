# Flexible Variational Graph Auto-Encoders

### Alex Kan and Armaan Sethi

## [Project Final Report](https://github.com/akan72/comp790/blob/master/reports/KanSethiFinalReport.pdf)  

### [Project Presentation Slides](https://docs.google.com/presentation/d/1Dv_icEAc2l0WAoKMwzf7syrZrpx9stSsgi613fbB2RQ/edit?usp=sharing)  

#### Introduction (from the Project Final Report) 

In recent years, Graph Neural Networks (GNNs) have proved successful for the task of learning problems posed on non-grid structured data such as graphs, meshes, and point clouds. Recently, Kipf and Welling have proposed a Graph Convolutional Network (GCN) layer that can be utilized for many learning tasks on graph represented data, including semi-supervised and unsupervised learning. These layers have the ability to learn flexible graph structure on the local level, and also can incorporate information gleaned from node features.  
  
  
There has been much previous work on deep learning for graph-structured data. Graphs that represent objects and their relationships are ubiquitous in the real world. Social networks, e-commerce networks, biological networks may all be represented using graphs. Graphs are usually complicated structures that contain rich underlying value.  
  
  
Graph data can be extremely varied and contain diverse structures. There are many different prop- erties of a graph that are important to consider, that are not represented in other data types. For example, graphs can be heterogeneous or homogeneous, weighted or unweighted, and signed or unsigned. Additionally, representation learning tasks on graphs can be either graph-focused, such as classification and generation, or node focused such as node classification and link prediction. These different tasks need different model architectures. Recently many new unsupervised methods such as Graph Autoencoders (GAEs), Graph Recurrent Neural Networks (Graph RNNs) and Graph Reinforcement learning have been used in order to learn the representation of graphs.  
  
  
Our work is currently focused on the link prediction problem with the realm of graph-structured data. We have chosen to expand on the Variational Autoencoder (VAE) framework with the goal of developing a flexible model that can learn a good latent representation of undirected graphs and accurately predict their edges
