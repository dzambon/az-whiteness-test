import numpy as np
import scipy.stats
from collections import namedtuple

def sum_duplicates(edge_index, edge_weight=None, inplace=False):
    """
    Adapted from scipy.sparse.coo_array.sum_duplicates.

    It takes the list of edges and removes the repeated ones.
    The weights associated with duplicated edges are summed together.
    """
    if not inplace:
        import copy
        edge_index_ = copy.deepcopy(edge_index)
        edge_weight_ = copy.deepcopy(edge_weight)
    edge_weight_is_array = isinstance(edge_weight_, np.ndarray)
    if edge_weight_is_array:
        assert edge_weight_.shape[0] == edge_index_.shape[1]

    if edge_index_.shape[1] == 0:
        return edge_index_, edge_weight_

    edge_index_.sort(axis=0)
    # row, col = edge_index_[0], edge_index_[1]
    order = np.lexsort(edge_index_)
    # row = row[order]
    # col = col[order]
    edge_index_ = edge_index_[:, order]
    if edge_weight_is_array:
        edge_weight_ = edge_weight_[order]
    # unique_mask = ((row[1:] != row[:-1]) |
    #                (col[1:] != col[:-1]))
    unique_mask = np.any(edge_index_[:, 1:] != edge_index_[:, :-1], axis=0)
    unique_mask = np.append(True, unique_mask)
    # row = row[unique_mask]
    # col = col[unique_mask]
    edge_index_ = edge_index_[:, unique_mask]
    if edge_weight_is_array:
        unique_inds, = np.nonzero(unique_mask)
        # data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        edge_weight_ = np.add.reduceat(edge_weight_, unique_inds, dtype=edge_weight_.dtype)
    return edge_index_, edge_weight_

def twosided_std_gaussian_pval(stat):
    return 2 * (1 - scipy.stats.norm.cdf(np.abs(stat)))

def adj_to_edge_list(A):
    assert A.ndim == 2
    edge_index_spatial = np.array(np.where(A))
    edge_weight_spatial = None
    if np.any(np.logical_and(A != 0, A != 1)):
        edge_weight_spatial = A[edge_index_spatial[0], edge_index_spatial[1]]
    return edge_index_spatial, edge_weight_spatial

def masked_median(x, mask):
    """
    Median of numpy array x with masked values.
    Where mask is True, the value is considered, where False it is discarded.
    Median is computed for each component in the feature dimension -1 (that is x=(T, N, F))
    """
    if mask is None:
        return np.median(x, axis=range(x.ndim-1), keepdims=True)
    else:
        assert np.all(x.shape == mask.shape)
        med = [np.median(x[..., f:f+1][mask[..., f:f+1].astype(bool)]) for f in range(x.shape[-1])]
        for i in range(x.ndim-1):
            med = [med]
        return np.array(med)

AZWhitenessTestResult = namedtuple('AZWhitenessTestResult', ('statistic', 'pvalue'))
AZWhitenessMultiTestResult = namedtuple('AZWhitenessMultiTestResult', ('statistic', 'pvalue', 'componentwise_tests'))

def az_whiteness_test(x, mask=None, multivariate=None, remove_median=False, **kwargs):
    """
    AZ-whiteness test.
    Wrapper of _az_whiteness_test dealing with graph signals with
    feature dimension greater than 1.
    """

    if remove_median:
        # be careful that when the estimated median is not
        # accurate it can lead to false alarm; this happens
        # for example when T=1, N<<100, F>>10
        x_median = masked_median(x=x, mask=mask)
        x -= x_median

    F = x.shape[-1]
    if F > 1:
        assert multivariate is not None, "When F>1, it must be specified if the test should be multivariate or not."
    else:
        multivariate = True

    if multivariate:
        return _az_whiteness_test(x=x, mask=mask, **kwargs)
    else:
        res = []
        for f in range(x.shape[-1]):
            x_ = x[..., f:f + 1]
            if mask is None:
                mask_ = None
            else:
                mask_ = mask[..., f:f + 1]
            res.append(_az_whiteness_test(x=x_, mask=mask_, **kwargs))
            C_multi = np.sum([r.statistic for r in res]) / np.sqrt(len(res))
            pval = twosided_std_gaussian_pval(C_multi)
        return AZWhitenessMultiTestResult(C_multi, pval, res)

def _to_numpy(o):
    if isinstance(o, np.ndarray):
        return o
    if isinstance(o, list):
        return np.array(o)
    if isinstance(o, int) or isinstance(o, float):
        return float(o)
    if o is None:
        return o
    import torch
    if isinstance(o, torch.Tensor):
        return o.numpy()
    raise NotImplementedError(f"I don't know how to convert {type(o)} to numpy")

def _az_whiteness_test(x: np.ndarray, edge_index_spatial: np.ndarray,
                       edge_weight_spatial: [np.ndarray, float, None] = None,
                       edge_weight_temporal: [float, None] = None,
                       mask: [np.ndarray, None] = None,
                       lamb: float = 0.5):
    """
    AZ-test for whiteness
    For temporal graphs,
        Ctilde(x, G) = Ctilde_spatial(x, G, Ws) + Ctilde_temporal(x, Wt)
    with
        Ctilde_temporal(x, Wt) = Wt * sum_{t=2..T} sum_v sign(x[v, t].dot(x[v, t-1]))
    and
        Ctilde_spatial(x, Ws) = sum_t sum_{(u, v) in edge_index} Ws[(u, v)] * sign(x[u, t].dot(X[v, t])).

    We have that
        Ctilde_temporal(x, Wt) = Wt * sum_v sign(sum(x[:, 1:] * dot(x[:, :-1], axis=F)
            = Wt * sign(sum( x[:, 1:] * dot(x[:, :-1], axis=F).sum(axis=[N, T])
            = Wt * sign(xxt.sum(axis=F)).sum(axis=[N, T])
    with
        xxt = x[:, 1:] * x[:, :-1]
    of shape (N, T-1, F).

    For Ctilde_spatial, we distinguish the cases of static and dynamic graph.
    If time_segmentation is None, ie the graph is static, we call
        xxs = x[edge_index[0]] * x[edge_index[1]]
    with xxs of shape (E, T, F), and write
        Ctilde_spatial(x, Ws) = sum_t sum_{(u, v) in edge_index} Ws[(u, v)] * sign(x[u, t].dot(X[v, t])).
            = sum_t sum_{(u, v) in edge_index} Ws[(u, v)] * sign(xxs.sum(axis=F))
            = (Ws[..., None] * sign(xxs.sum(axis=F))).sum(axis=[E, T])
            = einsum("e, et -> 1", Ws, sign(xxs.sum(axis=F)))

    If the graph is not static, we call
        xxs = x[edge_index[0], time_segmentation] * x[edge_index[1], time_segmentation]
    with xx of shape (E, F), and where E incorporates the edges of all time steps, so
        Ctilde_spatial(x) = sum_t sum_{(u, v) in edge_index} sign(x[u, t].dot(X[v, t])).
            = sum_t sum_{(u, v) in edge_index} sign(xxs.sum(axis=F))
            = sign(xxs.sum(axis=F)).sum(axis=E)

    When x is masked then this is how we proceed:
        x = x * mask
    so that, if a node is missing, then s((u,v)) = 0, and at the end I only need to remove
    from W2 the appropriate number of weights. This should also appropriately behave when
    only single node features are missing, and not the entire node is disabled.

    :param x: (N, T, F) Graph signal
    :param mask: (N, T, F) mask of x
    :param edge_index_spatial: (2, E). Edges of the static (spatial) graph
    :param edge_weight_spatial: (E,) or (1,). Weights of the spatial edges
    :param edge_weight_temporal: number or "auto. Weight for the temporal edges
    :param time_segmentation: (E,) tells which time step each edge corresponds to.
        It is basically like the batch tensor
    :param directed: (bool, def=False)
    :param lamb: (float in [0.0, 1.0], def=0.5) lamb defines the convex combination
        of the test applied on the spatial topology (lamb==1.0) and the temporal
        dimension (lamb=0.0)
    :return:

    """

    # Check datatypes
    x = _to_numpy(x)
    mask = _to_numpy(mask)
    edge_index_spatial = _to_numpy(edge_index_spatial)
    edge_weight_spatial = _to_numpy(edge_weight_spatial)

    T_DIM, N_DIM, F_DIM = 0, 1, 2
    T, N, F = x.shape[T_DIM], x.shape[N_DIM], x.shape[F_DIM]

    # Parse mask
    if mask is None:
        mask = np.ones_like(x)
    mask = mask.astype(int)
    assert np.all(np.logical_or(mask == 0, mask ==1))
    # mask_node = mask.prod(axis=F_DIM)
    mask_node = mask.max(axis=F_DIM)
    #mask data
    x = x * mask

    # Parse spatial edges
    #no duplicated edges
    edge_index_spatial, edge_weight_spatial = sum_duplicates(edge_index=edge_index_spatial, edge_weight=edge_weight_spatial)
    #no self-loops
    no_self_loops = edge_index_spatial[0] != edge_index_spatial[1]
    edge_index_spatial = edge_index_spatial[:, no_self_loops]
    E_spatial_unmasked = edge_index_spatial.shape[1]

    # Parse spatial edge weight
    assert N == edge_index_spatial.max() + 1, "Is the input signal (T, N, F)?"
    if isinstance(edge_weight_spatial, np.ndarray):
        edge_weight_spatial = edge_weight_spatial[no_self_loops]
    elif edge_weight_spatial is None:
        edge_weight_spatial = np.ones(E_spatial_unmasked)
    elif isinstance(edge_weight_spatial, int) or isinstance(edge_weight_spatial, float):
        edge_weight_spatial = edge_weight_spatial * np.ones(E_spatial_unmasked)
    assert edge_weight_spatial.shape[0] == E_spatial_unmasked
    assert np.all(edge_weight_spatial > 0)
    # Following mask finds edges with both ending nodes available.
    # Indices of spatial edges are repeated the in the second element
    # of the output because the corresponds to different time steps:
    #   mask_node (T, N) -> mask_edge_spatial ((time step,), (edge_spatial,)
    mask_edge_spatial = np.where(np.logical_and(
                             mask_node[:, edge_index_spatial[0]],
                             mask_node[:, edge_index_spatial[1]]))
    #sums over all non masked edges (it considers already the dynamic graph with all "repeated" edges)
    W2_spatial = np.sum(edge_weight_spatial[mask_edge_spatial[1]]**2)

    # Parse temporal edge weight
    if T > 1:
        assert T_DIM == 0
        E_temporal_masked = (mask[1:] * mask[:-1]).sum()
        if edge_weight_temporal == "auto" or edge_weight_temporal is None:
            edge_weight_temporal = np.sqrt(W2_spatial / E_temporal_masked)
    else:
        E_temporal_masked = 0
        edge_weight_temporal = 1
    assert isinstance(edge_weight_temporal, int) or isinstance(edge_weight_temporal, float)
    assert edge_weight_temporal > 0
    W2_temporal = (edge_weight_temporal ** 2) * E_temporal_masked

    # Inner products
    xxs = x[:, edge_index_spatial[0]] * x[:, edge_index_spatial[1]]  # (T, E, F) * (T, E, F) -> (T, E, F)
    xxs = xxs.sum(axis=F_DIM)                                  # (T, E, F) -> (T, E)
    assert T_DIM == 0
    xxt = x[1:] * x[:-1]   # (T-1, N, F) * (T-1, N, F) -> (T-1, N, F)
    xxt = xxt.sum(axis=F_DIM)       # (T-1, N, F) -> (T-1, N)

    # Weighted signs and Ctilde
    w_sgn_xxs = edge_weight_spatial[None, ...] * np.sign(xxs)  # (1, E) * (T, E) -> (T, E)
    Ctilde_spatial = w_sgn_xxs.sum()
    sgn_xxt = np.sign(xxt)
    Ctilde_temporal = edge_weight_temporal * sgn_xxt.sum()

    # Normalize Ctilde: C
    assert 0 <= lamb <= 1
    Ctilde = lamb * Ctilde_spatial + (1-lamb) * Ctilde_temporal
    W2 = (lamb**2) * W2_spatial + ((1-lamb)**2) * W2_temporal
    C = Ctilde / np.sqrt(W2)

    pval = twosided_std_gaussian_pval(C)
    return AZWhitenessTestResult(C, pval)

def optimality_check(x, mask=None,
                     edge_index_spatial=None, edge_weight_spatial=None, A=None,
                     **kwargs):

    from scipy.stats import ttest_1samp, binomtest, wilcoxon

    if not isinstance(x, np.ndarray):
        x = x.numpy()

    # Parse graph
    if A is not None:
        edge_index_spatial, edge_weight_spatial = adj_to_edge_list(A)

    # Parse mask
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    elif mask.dtype != bool:
        mask = mask.astype(bool)

    msg = []
    msg.append(f"Optimality of the forecasting model:")
    x_ravel = x[mask]

    msg.append(f" - [median=0, est={np.median(x_ravel):.3f}] {binomtest(k=np.sum(x_ravel>0), n=x_ravel.size)}")
    msg.append(f" - [mean=0, est={np.mean(x_ravel):.3f}] {ttest_1samp(a=x_ravel, popmean=0.0)}")
    msg.append(f" - [simmetry] {wilcoxon(x_ravel[::5])} (downsampled!)")

    lamb_passed = kwargs.pop("lamb", 0.0)
    for lamb in sorted(set([0.0, 0.5, 1.0] + [lamb_passed])):
        az_res = az_whiteness_test(x=x, mask=mask,
                                   edge_index_spatial=edge_index_spatial,
                                   edge_weight_spatial=edge_weight_spatial,
                                   lamb=lamb,
                                   **kwargs)
        if isinstance(az_res, list):
            for i, res in enumerate(az_res):
                msg.append(f" - [whiteness, lamb={lamb}, feat={i}] {res}")
        else:
            msg.append(f" - [whiteness, lamb={lamb}] {az_res}")

    # for m in msg:
    #     print(m)
    return msg

AVAILABLE_DISTRIBUTIONS = ["norm", "chi2(1)", "chi2(5)", "bi-norm", "chi2(1)-chi2(5)", "bi-unif14"]
def gen_nullmedian_signal(shape: tuple, distrib: str="norm"):
    """
    Generates iid graph signals from different distributions

    :param shape: (3-tuple) generally in the format (T, N, F)
    :param distrib: (str) identifier of the type of distribution
    :return:
    """
    if distrib == "norm":
        x = np.random.randn(*shape)
    elif distrib == "chi2(1)":
        import scipy.stats
        x = scipy.stats.chi2(df=1).rvs(size=shape)
        x -= scipy.stats.chi2(df=1).ppf(0.5)
    elif distrib == "chi2(5)":
        import scipy.stats
        x = scipy.stats.chi2(df=5).rvs(size=shape)
        x -= scipy.stats.chi2(df=5).ppf(0.5)
    # elif distrib == "mix":
    #     import scipy.stats
    #     mask = np.random.rand(*shape) > .7
    #     x = mask * scipy.stats.chi2(df=1).rvs(size=shape)
    #     x += (1 - mask) * scipy.stats.chi2(df=5).rvs(size=shape)
    elif distrib == "bi-norm":
        import scipy.stats
        mask = np.random.rand(*shape) > .5
        x = mask * np.random.randn(*shape)+3.
        x += (1 - mask) * np.random.randn(*shape)-3.
    elif distrib == "chi2(1)-chi2(5)":
        import scipy.stats
        mask = np.random.rand(*shape) > .5
        x = mask * scipy.stats.chi2(df=1).rvs(size=shape)
        x += (1 - mask) * scipy.stats.chi2(df=5).rvs(size=shape) * (-1.)
    elif distrib == "bi-unif14":
        import scipy.stats
        mask = np.random.rand(*shape) > .5
        x = mask * np.random.rand(*shape)
        x += (1 - mask) * np.random.rand(*shape) * (-4.)
    else:
        raise NotImplementedError(f"Distribution {distrib} is not available")

    return x

def gen_correlated_signal(x, G, c_space, c_time):
    from einops import rearrange
    x_ = G.adj.dot(rearrange(x, "T N F -> N (T F)"))
    x_new = x + c_space * rearrange(x_,  "N (T F) -> T N F", F=x.shape[-1])
    # x_new[2:] += c_time * (x_new[1:-1] + x_new[:-2])
    edge_den = G.num_edges / G.num_nodes
    x_new[1:] += c_time * edge_den * x_new[:-1]
    return x_new

def test_shuffle_dim(x, mask, **kwargs):

    T, N, F = x.shape
    pn = np.random.permutation(N)
    pt = np.random.permutation(T)

    print("---test 3 -----")
    for lamb in [0.5, 0.0, 1.0]:

        # optimality_check(x=x, mask=mask, lamb=0.5, **args)

        print(f"--- lambda {lamb} -----")
        az_res = az_whiteness_test(x=x, mask=mask, lamb=lamb, **kwargs)
        print(" original", lamb, az_res)

        az_res = az_whiteness_test(x=x[:, pn], mask=mask[:, pn], lamb=lamb, **kwargs)
        print("node perm", lamb, az_res)

        az_res = az_whiteness_test(x=x[pt], mask=mask[pt], lamb=lamb, **kwargs)
        print("time perm", lamb, az_res)
