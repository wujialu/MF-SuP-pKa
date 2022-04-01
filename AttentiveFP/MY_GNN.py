'''read out just used (super node||node)'''
import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score, mean_absolute_error
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
        g.ndata['hv_new'] = self.project_node(node_feats)
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


class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.
        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)


class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP
    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    This class computes graph representations out of node features.
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
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.
        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
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
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False, get_node_feats=False):
        """Graph-level regression/soft classification.
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
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        if get_node_feats:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_feats
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats)


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
    smiles, g_attentivefp, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    num_pos = 0
    num_impos = 0
    for i in labels:
        if i == 1:
            num_pos = num_pos + 1
        if i == 0:
            num_impos = num_impos + 1
    task_pos_weight = num_impos / (num_pos+0.00000001)
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
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))  # the input for F.mse_loss must be tensor
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
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
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
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
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
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
        a SMILES, a DGLGraph, all-task labels and descriptora
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    descriptors:
        descriptors used as global feature
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    smiles, g_attentivefp, labels, mask = map(list, zip(*data))
    attentivefp_bg = dgl.batch(g_attentivefp)
    # bg.set_n_initializer(dgl.init.zero_initializer)
    # bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)

    return smiles, attentivefp_bg, labels,  mask


def run_a_train_epoch(args, model, data_loader, loss_criterion, optimizer, task_weight=None):
    model.train()
    train_meter = Meter()
    total_loss = 0
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])
    for batch_id, batch_data in enumerate(data_loader):
        smiles, attentivefp_bg, labels, mask = batch_data
        attentivefp_bg = attentivefp_bg.to(args['device'])
        mask = mask.unsqueeze(dim=1).float().to(args['device'])
        labels = labels.unsqueeze(dim=1).float().to(args['device'])

        attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
        attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

        logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats)
        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if task_weight is None:
            loss = (loss_criterion(logits, labels)*(mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_criterion(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        total_loss = total_loss + loss * len(smiles)
        optimizer.step()
        # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, mask)
        del mask, labels,\
            attentivefp_bg, loss, logits
        torch.cuda.empty_cache()
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    return train_score, total_loss / 999


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, attentivefp_bg, labels, mask = batch_data
            attentivefp_bg = attentivefp_bg.to(args['device'])
            mask = mask.unsqueeze(dim=1).float().to(args['device'])
            labels = labels.unsqueeze(dim=1).float().to(args['device'])

            attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

            logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats)
            labels = labels.type_as(logits).to(args['device'])
            # Mask non-existing labels
            eval_meter.update(logits, labels, mask)
            del mask, labels,\
                attentivefp_bg, attentivefp_edge_feats, attentivefp_node_feats, logits
            torch.cuda.empty_cache()
        return eval_meter.compute_metric(args['metric_name'])[0]


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
    sensitivity = round(tp/(tp+fn), 4)
    specificity = round(tn/(tn+fp), 4)
    return sensitivity, specificity


def run_an_eval_epoch_detail(args, model, data_loader, out_path=None):
    model.eval()
    eval_meter = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, attentivefp_bg, labels, mask = batch_data
            attentivefp_bg = attentivefp_bg.to(args['device'])
            mask = mask.unsqueeze(dim=1).float().to(args['device'])
            labels = labels.unsqueeze(dim=1).float().to(args['device'])

            attentivefp_node_feats = attentivefp_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            attentivefp_edge_feats = attentivefp_bg.edata.pop(args['edge_data_field']).float().to(args['device'])

            logits = model(attentivefp_bg, attentivefp_node_feats, attentivefp_edge_feats)
            labels = labels.type_as(logits).to(args['device'])
            eval_meter.update(logits, labels, mask)

            smiles_list += smiles

            del mask, labels,\
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
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='../model/AttentiveFP/{}_early_stop.pth'.format(task_name)

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

    def step(self, score, model): #check higher
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