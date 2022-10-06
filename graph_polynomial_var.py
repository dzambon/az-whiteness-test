import torch
from torch_geometric.nn import MessagePassing
from tsl.utils import ArgParser
from einops import rearrange

from triangular_tricom_graph import TriCommunityGraph


def gcn_gso(edge_index, num_nodes):
    """
    Graph shift operator based on the adjacency matrix (passed as input) as done in
    the GCN paper: Kipf, Thomas N., and Max Welling. "Semi-supervised classification
    with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).
    See also [GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/graphgym.html?highlight=gcn#torch_geometric.graphgym.models.GCNConv)
    in PyG.
    """
    from torch_geometric.utils import add_self_loops, degree
    edge_index, _ = add_self_loops(edge_index=torch.tensor(edge_index), num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, edge_weight


class GraphPolyVARFilter(MessagePassing):
    """
    Polynomial spatiotemporal graph filter.
    For x in (nodes, time_steps), the filter is given by
    for t in range(T):
        x[:, t] = eps[:, t] + sum_{p=1}^P  sum_{l=0}^L  psi[l, p] * S**l . x[:, t-p]
    where
     - eps is some noise component
     - S is a graph shift operator (GSO), and
     - psi are the filter coefficients with L-hop neighbors, and P steps in the past

    See Eq. 13, Isufi, Elvin, et al. "Forecasting time series with varma recursions
    on graphs." IEEE Transactions on Signal Processing 67.18 (2019): 4870-4885.
    """

    def __init__(self,
                 spatial_order: int = None,
                 temporal_order: int = None,
                 node_feature_dim: int = 1,
                 filter_coefs = None,       # (L, P)
                 horizon: int = 1,
                 activation="tanh",
                 **kwargs):
        super().__init__(aggr="add", node_dim=-2)

        self.node_feature_dim = node_feature_dim
        if self.node_feature_dim != 1:
            raise NotImplementedError()

        if filter_coefs is not None:
            self.temporal_order = filter_coefs.shape[1]    # P
            self.spatial_order = filter_coefs.shape[0]     # L
            # x: (N, P) . (P, L) -> (N, L)
            self.lin_filter = torch.nn.Linear(in_features=self.temporal_order * self.node_feature_dim,
                                              out_features=self.spatial_order, bias=False)
            self.lin_filter.weight.requires_grad = False
            self.lin_filter.weight = torch.nn.Parameter(filter_coefs, requires_grad=False)

        else:
            self.temporal_order = temporal_order
            self.spatial_order = spatial_order
            self.lin_filter = torch.nn.Linear(in_features=self.temporal_order * self.node_feature_dim,
                                              out_features=self.spatial_order, bias=False)

        assert self.temporal_order is not None
        assert self.spatial_order is not None

        self.activation = activation
        self.horizon = horizon

    def forward(self, x, edge_index, edge_weight=None):
        """

        :param x: (B, T>=P, N, F=1)
        :param edge_index: (2, E)
        :param edge_weight: (E,)
        :return:
        
        .............
        x has shape [N, in_channels]
        edge_index has shape [2, E]

        h = x[:, t-P: t-1] . psi.T    # (N, P) . (P, L) -> (N, L)
        for l = 1 ... L:
            h[:, l:] = S . h[:, l:]
        x[t] = h.sum(axis=1)

        ...................

        x[:, t] = eps[:, t] + sum_{p=1}^P  sum_{l=0}^L  psi[l, p] * S**l . x[:, t-p]
            = eps[:, t] + sum_{l=0}^L  sum_{p=1}^P  psi[l, p] * S**l . x[:, t-p]
            = eps[:, t] + sum_{l=0}^L  S**l . (x[:, t-P: t-1] . psi[l, :])

        h = x[:, t-P: t-1] . psi.T    # (N, P) . (P, L) -> (N, L)
        for l = 1 ... L:
            h[:, l:] = S . h[:, l:]
        x[t] = eps + h.sum(axis=1)

        """
        has_no_batch = False
        if x.ndim == 3:  # (T, N, F)
            has_no_batch = True
            x = x.unsqueeze(0)
        assert self.horizon == 1
        assert x.shape[-3] >= self.temporal_order  # time steps
        assert edge_index.max() + 1 == x.shape[-2] # number of nodes
        assert x.shape[-1] == 1                    # node features

        # (B, T>=P, N, F=1) -> (B, N, P)
        x_ = rearrange(x[:, -self.temporal_order:], "B P N F -> B N (P F)")
        # x_ = x[-self.temporal_order:].reshape(self.temporal_order, -1).transpose(1, 0)
        # (B, N, P) -> (B, N, L)
        h = self.lin_filter(x_)  #.transpose(1, 0).unsqueeze(-1)
        for l in range(self.spatial_order):
            h[..., l+1:] = self.propagate(edge_index=edge_index, x=h[..., l+1:], norm=edge_weight)
        # (B, N, L) -> (B, T=1, N, F=1)
        x_ = h.sum(axis=-1).unsqueeze(-2).unsqueeze(-1)
        if self.activation == "global":
            x_ /= torch.abs(x_).max()
        elif self.activation == "tanh":
            x_ = torch.tanh(x_)
        elif self.activation is not None:
            x_ = self.activation(x_)

        if has_no_batch:
            assert x_.shape[0] == 1
            return x_[0]
        else:
            return x_

    def predict(self, x, **kwargs):
        T, N, F = x.shape
        assert F == 1
        y_true = x.reshape(T, N, F)
        y_pred = torch.empty((T, N, F))
        for t in range(self.temporal_order, T):
            y_pred[t: t+1] = self.forward(y_true[t - self.temporal_order: t], **kwargs)
        return y_pred[self.temporal_order:], y_true[self.temporal_order:]

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--spatial-order', type=int, default=2)
        parser.opt_list('--temporal-order', type=int, default=3)
        parser.opt_list('--activation', type=str, default="tanh")
        return parser


class GraphPolyVARDataset(object):
    """
    A synthetic dataset generated from a GraphPolyVarFilter filter on a
    TriCommunityGraph graph.
    """

    _path_default = "./data/gpvardata"

    def __init__(self, coefs, sigma_noise=.2, communities=5, name=None, **kwargs):
        if name is None:
            self.name = f"GP-VAR-TriComm[c{communities}]"
        else:
            self.name = name
        self.G = TriCommunityGraph(communities=communities, **kwargs)
        self.gso = gcn_gso(self.G.edge_index, self.G.num_nodes)

        self.coefs = coefs
        self.filter = GraphPolyVARFilter(filter_coefs=coefs, **kwargs)
        self.sigma_noise = sigma_noise

        self.mask = None
        self.x = None

    @property
    def mae_optimal(self):
        """ E[|X|] of a Gaussian X"""
        return torch.sqrt(torch.tensor(2.0 / 3.14159265359)) * self.sigma_noise

    @classmethod
    def load_dataset(cls, path=None):
        import os.path

        if path is None:
            path = cls._path_default

        file_to_read = open(os.path.join(os.path.abspath(path), "dataset.pickle"), "rb")
        import pickle
        loaded_object = pickle.load(file_to_read)
        file_to_read.close()

        return loaded_object

    def dump_dataset(self, path=None):
        import os.path

        if path is None:
            path = self._path_default + f"-T{self.x.shape[0]}_{self.G.community_connectivity}-c{self.G.num_communities}"

        path_unique = os.path.abspath(path)
        ct = 0
        while os.path.isdir(path_unique):
            ct += 1
            path_unique = os.path.abspath(path + f"({ct})")
        os.makedirs(path_unique)
        data_file = os.path.join(path_unique, "dataset.pickle")
        file_to_store = open(data_file, "wb")
        import pickle
        pickle.dump(self, file_to_store)
        file_to_store.close()

        return path_unique

    def generate_data(self, T: int):
        """

        :param T: num of time steps
        :return: data x in (T, N, F=1)
        """
        N = self.G.num_nodes
        F = 1
        P = self.filter.temporal_order

        x = torch.empty((T, N, F))
        x[:P] = torch.randn((P, N, F))
        eps = torch.randn(x.shape) * self.sigma_noise

        edge_index, edge_weight = self.gso
        for t in range(P, T):
            x[t: t+1] = self.filter.forward(x[t - P: t], edge_index=edge_index, edge_weight=edge_weight) + eps[t: t+1]

        self.x = x
        self.mask = torch.ones_like(self.x)
        return self.x

    def numpy(self, return_idx=False):
        if return_idx:
            return self.x.numpy(), range(self.x.shape[0])
        else:
            return self.x.numpy()

    def get_splitter(self, val_len, test_len):
        from tsl.data.datamodule.splitters import TemporalSplitter
        return TemporalSplitter(val_len=val_len, test_len=test_len)

