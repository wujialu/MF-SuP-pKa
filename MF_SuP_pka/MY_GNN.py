'''read out just used (super node||node)'''
import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score, \
    mean_absolute_error
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl import function as fn
from dgl.nn.pytorch.conv import RelGraphConv, GATConv
from dgl.nn.pytorch import edge_softmax
from sklearn import metrics
from torch import nn
import pandas as pd
import torch as th
from dgllife.model.gnn import gcn
from tqdm import tqdm
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


# pylint: disable=W0221, C0103, E1101
class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.
    This will be used in GNN layers for updating node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.
    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)  # 40==》200
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    """GNNLayer for updating node features.
    This layer performs message passing over node representations and update them.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation.
        Generate edge features by concatenating the features of the destination
        and source nodes.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.
        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)


class AttentiveFPGNN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    This class performs message passing in AttentiveFP and returns the updated node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPGNN, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats


class SubgraphPool(nn.Module):
    """One-step readout in MF_SuP_pka
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, feat_size, dropout, attenuation_lambda=0.1, distance_matrix_kernel='exp'):
        super(SubgraphPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

        # self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))
        #
        # if distance_matrix_kernel == 'softmax':
        #     self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        # elif distance_matrix_kernel == 'exp':
        #     self.distance_matrix_kernel = lambda x: torch.exp(-x)

    def forward(self, g, node_feats, subgraph_feats, get_node_weight=False):
        """Perform one-step readout
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs. The number of graphs is equal to the number of ionization centers in a batch.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        subgraph_feats : float32 tensor of shape (SG, graph_feat_size)
            Input graph features. SG for the number of subgraphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.
        Returns
        -------
        float32 tensor of shape (SG, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['m'] = torch.cat([dgl.broadcast_nodes(g, F.relu(subgraph_feats)), node_feats],
                                     dim=1)  # (num_nodes, 400)

            g.ndata['z'] = self.compute_logits(g.ndata['m'])  # (num_nodes, 1)

            g.ndata['a'] = dgl.softmax_nodes(g, 'z')  # (num_nodes, 1)

            g.ndata['hv'] = self.project_nodes(node_feats)  # (num_nodes, 200)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')  # (batch, 200)
            context = F.elu(g_repr)  # (batch, 200)

            if get_node_weight:
                return self.gru(context, subgraph_feats), g.ndata['a']
            else:
                return self.gru(context, subgraph_feats)


def extract_k_hop_subgraph(bg_list, num_sg, k):
    sg_list = []

    for i, g in enumerate(bg_list):
        num_sg_i = num_sg[i].item()
        use_adj = g.ndata['use_adj']  # (num_nodes, 10)
        # print(use_adj)
        for j in range(num_sg_i):
            nodes_mask = torch.tensor([False] * use_adj.shape[0])
            adj = use_adj[:, j]
            node_idx = (adj <= k).nonzero()
            nodes_mask[node_idx] = True
            sg = dgl.node_subgraph(g, nodes_mask)
            sg.ndata['weight'] = torch.exp(-0.1 * adj[node_idx]).float()
            sg_list.append(sg)
    assert len(sg_list) == torch.sum(num_sg)
    return sg_list


def extract_ego_graph(bg_list, num_sg):
    ego_g_list = []

    for i, g in enumerate(bg_list):
        num_sg_i = num_sg[i].item()
        for j in range(num_sg_i):
            ego_g_list.append(g)
    assert len(ego_g_list) == torch.sum(num_sg)
    return ego_g_list


def micro_to_macro_predict(new_bg, acid_or_base=None):
    if acid_or_base == 'acid':
        new_bg.ndata['micro_pka'] = torch.pow(10, -new_bg.ndata['micro_pka'])
        macro_pka = - torch.log10(dgl.sum_nodes(new_bg, 'micro_pka'))
    elif acid_or_base == 'base':
        new_bg.ndata['micro_pka'] = torch.pow(10, new_bg.ndata['micro_pka'])
        macro_pka = torch.log10(dgl.sum_nodes(new_bg, 'micro_pka'))
    return macro_pka


class SuP_pka_Readout(nn.Module):
    """Readout phase in MF_SuP_pka.
    This class computes subgraph representations out of node features.
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(SuP_pka_Readout, self).__init__()

        self.subgraph_readouts = nn.ModuleList()

        for i in range(num_timesteps):
            self.subgraph_readouts.append(SubgraphPool(feat_size, dropout))

    def forward(self, bg, node_feats, num_sg, k=0, get_node_weight=False):
        """Computes subgraph representations out of node features.
        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.
        num_sg: float32 tensor of shape (B, 1)
            The number of subgraphs.
        k: int
            The radius of the subgraph. The optimal values are 2 for acidic task and 3 for basic task.
        Returns
        -------
        g_feats : float32 tensor of shape (SG, graph_feat_size)
            Subgraph representations computed. SG for the number of subgraphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with bg.local_scope():
            bg.ndata['hv'] = node_feats  # (num_nodes, 200)

            bg_list = dgl.unbatch(bg)  # a list of graphs
            sg_list = extract_k_hop_subgraph(bg_list, num_sg, k)  # a list of subgraphs
            sg_batch = dgl.batch(sg_list)
            sg_feats = dgl.sum_nodes(sg_batch, 'hv', 'weight')

            ego_g_list = extract_ego_graph(bg_list, num_sg)  # a expanded list of graphs, num_graphs=num_subgraphs
            ego_g_batch = dgl.batch(ego_g_list)
            # print(len(ego_g_list))
            ego_g_node_feats = ego_g_batch.ndata['hv']

        if get_node_weight:
            node_weights = []

        for subgraph_readout in self.subgraph_readouts:
            if get_node_weight:
                sg_feats, node_weights_t = subgraph_readout(ego_g_batch, ego_g_node_feats, sg_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                sg_feats = subgraph_readout(ego_g_batch, ego_g_node_feats, sg_feats)

        if get_node_weight:
            return sg_feats, node_weights
        else:
            return sg_feats


class SuP_pka_Predictor(nn.Module):
    """SuP_pka is a variant of AttentiveFP with subgraph pooling mechanism.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    acid_or_base: str
        The endpoint: acidic pka or basic pka.
    k: int
        The radius of the subgraph. The optimal values are 2 for acidic task and 3 for basic task.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.,
                 acid_or_base=None,
                 k=0):
        super(SuP_pka_Predictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)

        self.readout = SuP_pka_Readout(feat_size=graph_feat_size,
                                       num_timesteps=num_timesteps,
                                       dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

        self.acid_or_base = acid_or_base
        self.k = k

    def forward(self, g, node_feats, edge_feats, num_sg, new_bg, get_node_weight=False, get_subgraph_embedding=False):
        """Subgraph-level regression.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.
        get_subgraph_embedding: bool
            Whether to get the updated subgraph embedding. Default to False.
        Returns
        -------
        micro_pka: float32 tensor of shape (num_sg, 1).
        macro_pka: float32 tensor of shape (G, 1). G for the number of graphs in a batch.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        subgraph_embedding: float32 tensor of shape (num_sg, 1), optional
            This is returned when ``get_subgraph_embedding`` is ``True``.
        """

        node_feats = self.gnn(g, node_feats, edge_feats)

        if get_node_weight:
            g_feats, node_weight = self.readout(g, node_feats, num_sg, self.k, get_node_weight)
            micro_pka = self.predict(g_feats)
            new_bg.ndata['micro_pka'] = micro_pka
            macro_pka = micro_to_macro_predict(new_bg, acid_or_base=self.acid_or_base)
            return micro_pka, macro_pka, node_weight
        else:
            g_feats = self.readout(g, node_feats, num_sg, self.k, get_node_weight)
            micro_pka = self.predict(g_feats)
            new_bg.ndata['micro_pka'] = micro_pka
            macro_pka = micro_to_macro_predict(new_bg, acid_or_base=self.acid_or_base)
            if get_subgraph_embedding:
                return micro_pka, macro_pka, g_feats
            else:
                return micro_pka, macro_pka


def set_random_seed(seed=10):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set):
    smiles, g_attentivefp, labels, mask, num_sg = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    num_pos = 0
    num_impos = 0
    for i in labels:
        if i == 1:
            num_pos = num_pos + 1
        if i == 0:
            num_impos = num_impos + 1
    task_pos_weight = num_impos / (num_pos + 0.00000001)
    task_pos_weight_list.append(task_pos_weight)
    return task_pos_weight_list


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))  # input and target must be tensor
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(
                metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and descriptors
        a binary mask indicating the existence of labels.
    Returns
    -------
    num_sg: Tensor with shape (B, 1).
        The number of subgraphs(ionization centers) for each molecule.
    new_bg: A batch of graph for converting micro-pka to macro-pka .
        The nodes in new graphs represent the k-hop subgraphs around the ionization centers.
    """
    smiles, g_attentivefp, labels, mask, num_sg = map(list, zip(*data))

    attentivefp_bg = dgl.batch(g_attentivefp)
    # bg.set_n_initializer(dgl.init.zero_initializer)
    # bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)
    num_sg = torch.tensor(num_sg)

    new_bg_list = []
    for i in range(len(smiles)):
        num_sg_nodes = num_sg[i].item()
        new_g = dgl.DGLGraph()
        new_g.add_nodes(num_sg_nodes)
        new_bg_list.append(new_g)
    new_bg = dgl.batch(new_bg_list)

    return smiles, attentivefp_bg, labels, mask, num_sg, new_bg


def run_a_train_epoch(args, model, data_loader, loss_criterion, optimizer, task_weight=None):
    model.train()
    train_meter = Meter()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, attentivefp_bg, labels, mask, num_sg, new_bg = batch_data

        attentivefp_bg = attentivefp_bg.to(args['device'])
        new_bg = new_bg.to(args['device'])

        mask = mask.unsqueeze(dim=1).float().to(args['device'])
        labels = labels.unsqueeze(dim=1).float().to(args['device'])
        num_sg = num_sg.unsqueeze(dim=1).to(args['device'])  # (G,1)
        attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
        attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

        _, logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats, num_sg, new_bg)
        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if task_weight is None:
            loss = (loss_criterion(logits, labels) * (mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_criterion(logits, labels) * (mask != 0).float(), dim=0) * task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        total_loss = total_loss + loss * len(smiles)
        optimizer.step()
        # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, mask)
        del mask, labels, \
            attentivefp_bg, loss, logits
        torch.cuda.empty_cache()
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))  # r2
    return train_score, total_loss / 999


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, attentivefp_bg, labels, mask, num_sg, new_bg = batch_data
            attentivefp_bg = attentivefp_bg.to(args['device'])
            new_bg = new_bg.to(args['device'])
            mask = mask.unsqueeze(dim=1).float().to(args['device'])
            labels = labels.unsqueeze(dim=1).float().to(args['device'])
            num_sg = num_sg.unsqueeze(dim=1).to(args['device'])  # (G,1)

            attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

            _, logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats, num_sg, new_bg)
            labels = labels.type_as(logits).to(args['device'])
            # Mask non-existing labels
            eval_meter.update(logits, labels, mask)
            del mask, labels, \
                attentivefp_bg, attentivefp_edge_feats, attentivefp_node_feats, logits
            torch.cuda.empty_cache()
        return eval_meter.compute_metric(args['metric_name'])


def sesp_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp = tp + 1
        if y_true[i] == y_pred[i] == 0:
            tn = tn + 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
    sensitivity = round(tp / (tp + fn), 4)
    specificity = round(tn / (tn + fp), 4)
    return sensitivity, specificity


def run_an_eval_epoch_detail(args, model, data_loader, out_path=None):
    model.eval()
    eval_meter = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, attentivefp_bg, labels, mask, num_sg, new_bg, hyper_edge_bg = batch_data

            attentivefp_bg = attentivefp_bg.to(args['device'])
            new_bg = new_bg.to(args['device'])

            mask = mask.unsqueeze(dim=1).float().to(args['device'])
            labels = labels.unsqueeze(dim=1).float().to(args['device'])
            num_sg = num_sg.unsqueeze(dim=1).to(args['device'])  # (G,1)

            attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

            _, logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats, num_sg, new_bg)
            labels = labels.type_as(logits).to(args['device'])
            eval_meter.update(logits, labels, mask)

            smiles_list += smiles

            del mask, labels, \
                attentivefp_edge_feats, attentivefp_node_feats, logits
            torch.cuda.empty_cache()
    prediction_pd = pd.DataFrame()
    y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    if args['task_type'] == 'classification':
        y_true_list = y_true.squeeze(dim=1).tolist()
        y_pred_list = torch.sigmoid(y_pred).squeeze(dim=1).tolist()
        # save prediction
        prediction_pd['smiles'] = smiles_list
        prediction_pd['label'] = y_true_list
        prediction_pd['pred'] = y_pred_list
        if out_path is not None:
            prediction_pd.to_csv(out_path, index=False)
        y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
        auc = metrics.roc_auc_score(y_true_list, y_pred_list)
        accuracy = metrics.accuracy_score(y_true_list, y_pred_label)
        se, sp = sesp_score(y_true_list, y_pred_label)
        pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label, zero_division=0)
        mcc = metrics.matthews_corrcoef(y_true_list, y_pred_label)
        f1 = f1[1]
        rec = rec[1]
        pre = pre[1]
        err = 1 - accuracy
        result = [auc, accuracy, se, sp, f1, pre, rec, err, mcc]
        return result
    else:
        y_true_list = y_true.squeeze(dim=1).tolist()
        y_pred_list = y_pred.squeeze(dim=1).tolist()
        # save prediction
        prediction_pd['smiles'] = smiles_list
        prediction_pd['label'] = y_true_list
        prediction_pd['pred'] = y_pred_list
        if out_path is not None:
            prediction_pd.to_csv(out_path, index=False)
        r2 = metrics.r2_score(y_true_list, y_pred_list)
        mae = metrics.mean_absolute_error(y_true_list, y_pred_list)
        rmse = (metrics.mean_squared_error(y_true_list, y_pred_list)) ** 0.5
        result = [r2, mae, rmse]
        return result


class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    taskname : str or None
        Filename for storing the model checkpoint

    """

    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None,
                 task_name='None'):
        if filename is None:
            task_name = task_name
            filename = '../model/AttentiveFP/{}_test_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = '../model/AttentiveFP/' + pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_model = torch.load(self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
