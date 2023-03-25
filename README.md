# MF-SuP-pKa
Pytorch implementation of “MF-SuP-pKa: multi-fidelity modeling with subgraph pooling mechanism for pKa prediction”
![image](https://github.com/wujialu/MF-SuP-pKa/blob/main/Graph%20abstract.png)
- **MF-SuP-pka** is a novel pka prediction model that utilizes subgraph pooling, multi-fidelity learning and data augmentation.
Compared with Attentive FP, MF-SuP-pKa achieves **23.83%** and **20.12%** improvement in terms of mean absolute error (MAE) on the DataWarrior acidic and basic sets, respectively.
## Overview 
- ```AttentiveFP/```: the implementation of Attentive FP for pka prediction.
- ```Graph_pka/```: the implementation of Graph-pKa.
- ```MF_SuP_pka/```: the source codes of MF-SuP-pKa.
- ```data/```: the pre-training data set, fine-tuning data set, and external test data sets used in MF-SuP-pKa.
- ```model/```: the pre-trained model weights of MF-SuP-pKa.
- ```prediction/```: the prediction results on SAMPL6, SAMPL7, and Novartis data sets by MF-SuP-pKa and other counterparts.
## Requirements
- python 3.6
- rdkit 2021.09.4
- sklearn 0.24.2
- torch 1.10.1
- dgl 0.6.0
## Usage
- Build graph data set<br>
```python build_pka_graph_dataset.py --dataset pka_acidic_2750 --type acid ```
- Train the MF-SuP-pKa model
  - Train from scratch<br>
  ```python MF_SuP_pka_model.py --task_name pka_acidic_2750 --type acid --k_hop 2 --stage before_transfer```
  - Fine-tuning the pre-trained model<br>
  ```python MF_SuP_pka_model.py --task_name pka_acidic_2750 --type acid --k_hop 2 --stage transfer --pretrain_aug```
## Reference
If you use this code, please cite the following paper:

Wu, J., Wan, Y., Wu, Z., Zhang, S., Cao, D., Hsieh, C. Y., & Hou, T. (2022). MF-SuP-pKa: multi-fidelity modeling with subgraph pooling mechanism for pKa prediction. Acta Pharmaceutica Sinica B.

Links: https://www.sciencedirect.com/science/article/pii/S2211383522004622
