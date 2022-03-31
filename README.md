# MF-SuP-pKa
Pytorch implementation of “MF-SuP-pKa: multi-fidelity modeling with subgraph pooling mechanism for pKa prediction”
![image](https://github.com/wujialu/MF-SuP-pKa/blob/main/Graph%20abstract.png)
- **MF_SuP_pka** is a novel pka prediction model that utilizes subgraph pooling, multi-fidelity learning and data augmentation.
Compared with Attentive FP, MF-SuP-pKa achieves **23.83%** and **20.12%** improvement in terms of mean absolute error (MAE) on the DataWarrior acidic and basic sets, respectively.
## Overview 
- ```data/```
- ```weights/```
- ```scripts/```
## Requirements
- rdkit 2021.09.4
- sklearn 0.24.2
- torch 1.10.1
- dgl 0.6.0
## Usage
