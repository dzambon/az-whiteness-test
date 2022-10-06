import os
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from graph_sign_test import az_whiteness_test
from triangular_tricom_graph import TriCommunityGraph

RESULT_FOLDER = "./results"

def run_simulation(signal_gen_fun, alpha, edge_index, repetitions):
    stats = []
    alarms = 0
    for r in range(repetitions):
        x = signal_gen_fun()
        aztest = az_whiteness_test(x=x, edge_index_spatial=edge_index, edge_weight_temporal="auto", multivariate=True)
        stats.append(aztest.statistic)
        if aztest.pvalue < alpha:
            alarms += 1
    return alarms, stats

def run_multiple_simulations(G, alpha, considered_edge_index,
                             F_list, T_list, cs_list, ct_list, distrib_list,
                             repetitions, disable_warning):
    results = []
    assert len(F_list) == len(T_list)
    import time
    elapsed = 0
    for distrib in distrib_list:
        for F, T in zip(F_list, T_list):
            for cs, ct in zip(cs_list, ct_list):
                corr = max([cs, ct])
                print(f"Test: F={F}, T={T}, corr={corr}.\tAlarm rate: ", end="")

                stats = []
                alarms = 0

                for r in range(repetitions):
                    if distrib == "norm":
                        subtract_median = False
                    else:
                        subtract_median = True
                        if not disable_warning:
                            raise Warning("Subtracting median")
                    x = gen_correlated_signals_fun(F=F, T=T, G=G,
                                                   corr_spatial=cs, corr_temporal=ct,
                                                   distrib=distrib, subtract_median=subtract_median)
                    t = time.time()
                    aztest = az_whiteness_test(x=x, edge_index_spatial=considered_edge_index, multivariate=False)
                    elapsed += time.time() - t

                    stats.append(aztest.statistic)
                    if aztest.pvalue < alpha:
                        alarms += 1

                alarm_rate = alarms / repetitions
                print(f"Alarm rate = {alarm_rate}\t({alarms}/{repetitions})")
                results.append((F, T, corr, alarm_rate, distrib))

                if repetitions == 1:
                    savefig=os.path.join(RESULT_FOLDER,f"gsignal_c{corr}")
                    G.plot(signal=x[..., 0], savefig=savefig + f"T{T}_{distrib}.pdf")
                    # plt.title(f"median = {np.median(x)}")
                    # plt.tight_layout()
                    # plt.savefig(savefig + f"topology.pdf")
                    print(f"empirical median {np.median(x)}")

                # import scipy.stats
                # s = np.array(stats)
                # scipy.stats.probplot(s, dist="norm", plot=plt)
                # plt.title(f"F{F} T{T} corr={corr} A{alarm_rate} mu{np.mean(s):.3f}+-{1.0/np.sqrt(repetitions):.3f} std{np.std(s):.3f}")
                # plt.show()

    runs = len(distrib_list)*len(F_list)*len(ct_list)*repetitions
    print("Elapsed times: ", elapsed, "for a total of ", runs, "runs")
    print("Average run time: ", elapsed/runs)

    df = pd.DataFrame(results, columns=["F", "T", "corr", "alarm_rate", "distrib"])
    df = df.astype(dict(F=int, T=int, corr=float, alarm_rate=float, distrib=str))
    return df, x

def plot_results(df, name, subplot_dist=True):

    F_list = df["F"].unique()
    distrib_list = df["distrib"].unique()

    # col_list, col_name = F_list, "F"
    # row_list, row_name = distrib_list, "distrib"
    # if col_list.size == 1 and row_list.size > 1:
    #     tmp = row_list
    #     row_list = col_list
    #     col_list = tmp

    if F_list.size > distrib_list.size:
        col_list, col_name = F_list, "F"
        row_list, row_name = distrib_list, "distrib"
    else:
        col_list, col_name = distrib_list, "distrib"
        row_list, row_name = F_list, "F"

    cols = col_list.size
    ax = None
    if subplot_dist:
        rows = row_list.size
        fig, ax = plt.subplots(figsize=(3*cols, 3*rows))
    else:
        rows = 1

    # for d, distrib in enumerate(distrib_list):
    for r, row_el in enumerate(row_list):
        if not subplot_dist:
            fig, ax = plt.subplots(figsize=(3*cols, 3*rows))
            r = 0

        # for i, F in enumerate(F_list):
        for c, col_el in enumerate(col_list):
            plt.subplot(rows, cols, r * cols + c + 1)

            extra_args = dict(annot=True, vmin=0, vmax=1)
            if r < rows-1:
                extra_args["xticklabels"] = []
            if c > 0:
                extra_args["yticklabels"] = []
            if c < cols-1:
                extra_args["cbar"] = False


            df_ = df.where(df[col_name] == col_el).where(df[row_name] == row_el)
            df_ = df_.dropna().astype(dict(corr=float, T=int))
            hm_ = sb.heatmap(df_.pivot("corr", 'T', "alarm_rate"),
                             **extra_args)
            # plt.axis("equal")

            if r < rows - 1:
                plt.xlabel("")
                # plt.ylabel("")
            if c > 0:
                plt.ylabel("")

            # tit = col_el if col_name == "distrib" else row_el
            if col_name == "distrib":
                distrib = col_el
                F = row_el
            else:
                distrib = row_el
                F = col_el

            # plt.rcParams.update({
            #     "text.usetex": True,
            #     "font.family": "Helvetica"
            # })
            if len(F_list) == 1 and F==1:
                plt.title(f"{DISTRIBUTIONS[distrib]}")
            else:
                plt.title(f"{DISTRIBUTIONS[distrib]} (F={F})")

        if not subplot_dist:
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_FOLDER, f'{name}_heatmap_alarmrate_{distrib}.pdf'))

    if subplot_dist:
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_FOLDER, f'{name}_heatmap_alarmrate.pdf'))
    plt.show()

def make_table(df):
    df_ = df.pivot_table(values="alarm_rate", index=['distrib', 'corr'], columns=["F", "T"])
    print(df_.to_latex())

def gen_correlated_signals_fun(F, T, G, corr_spatial, corr_temporal, distrib="norm", subtract_median=False):
    x = gen_nullmedian_signal(shape=(T, G.num_nodes, F), distrib=distrib)
    x = gen_correlated_signal(x=x, G=G, c_space=corr_spatial, c_time=corr_temporal)
    if subtract_median:
        x -= np.median(x)
    return x

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

def to_grid(list1, list2):
    return [l1 for l1 in list1 for _ in list2], \
           [l2 for _ in list1 for l2 in list2]


# DISTRIBUTIONS_UNIMODAL = {"norm":           "$\mathcal N(0,1)$",
#                           "chi2(1)":        "$\chi_2(1)$",
#                           "chi2(5)":        "$\chi_2(5)$"}
# DISTRIBUTIONS_MIXTURE = {"bi-norm":         "$\mathcal N(-3,1) + \mathcal N(3,1)",
#                          "chi2(1)-chi2(5)": "$\chi_2(1)-\chi_2(5)$",
#                          "bi-unif14":       "$U[-4,0)+U[0,1)"}
DISTRIBUTIONS_UNIMODAL = {"norm":           "N(0,1)",
                          "chi2(1)":        "chi2(1)",
                          "chi2(5)":        "chi2(5)"}
DISTRIBUTIONS_MIXTURE = {"bi-norm":         "N(-3,1) + N(3,1)",
                         "chi2(1)-chi2(5)": "chi2(1) - chi2(5)",
                         "bi-unif14":       "U[-4,0) + U[0,1)"}
DISTRIBUTIONS = {**DISTRIBUTIONS_UNIMODAL, **DISTRIBUTIONS_MIXTURE}

def main(experiment, show_signal=False, disable_warning=False):

    G = TriCommunityGraph(communities=3, connectivity="triangle")

    if show_signal:
        x = np.random.randn(G.num_nodes).reshape(-1, 1)
        G.plot(signal=x)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_FOLDER, "static-graph-signal.pdf"))
        plt.show()

    alpha = 0.05
    rep = 1000
    graphs = ["sparse"]
    distrib = ["norm"]
    F_list = [1, 2] #, 4, 8]
    T_list = [1, 10] #, 100, 1000]
    corr_spatial = [0.04, 0.16]
    corr_temporal = corr_spatial

    if experiment == "viz":
        distrib = ["bi-unif14"]
        distrib = ["chi2(1)"]
        distrib = ["norm"]
        F_list, T_list = to_grid([1], [500])
        corr_spatial = [0.0, 0.04, 0.16, 0.64]
        corr_temporal = corr_spatial
        rep = 1

    if experiment == "run-time":
        # rep = 100
        F_list, T_list = to_grid([1], [10000])
        corr_spatial = [0.01]
        corr_temporal = corr_spatial

    if experiment[:5] == "power":
        # Check detection rates for different distributions (regardless of the symmetry)
        if experiment == "power-unimodal":
            distrib = list(DISTRIBUTIONS_UNIMODAL.keys())
        elif experiment == "power-mixture":
            distrib = list(DISTRIBUTIONS_MIXTURE.keys())
        F_list, T_list = to_grid([1], [1, 10, 100, 1000])
        corr_spatial = [0.0, 0.01, 0.04, 0.16]
        corr_temporal = corr_spatial

    if experiment == "symmetry":
        # Check different distributions
        distrib = AVAILABLE_DISTRIBUTIONS
        F_list, T_list = to_grid([1, 4, 8], [10, 100, 1000])
        corr_spatial = [0.0]
        corr_temporal = corr_spatial

    if experiment == "sparse-full-only-time":
        # Compare sparse vs full
        F_list, T_list = to_grid([1, 4, 8], [1, 10, 100, 1000])
        # corr_spatial = [0.0, 0.01, 0.04, 0.16]
        # corr_temporal, corr_spatial = to_grid([0.0, 0.01, 0.04, 0.16], [0])
        corr_temporal, corr_spatial = to_grid([0], [0.0, 0.01, 0.04, 0.16])
        graphs = ["sparse", "complete"]

    if experiment == "sparse-full":
        # Compare sparse vs full
        F_list, T_list = to_grid([1, 4, 8], [1, 10, 100, 1000])
        # corr_spatial = [0.0, 0.01, 0.04, 0.16]
        # corr_temporal, corr_spatial = to_grid([0.0, 0.01, 0.04, 0.16], [0])
        corr_temporal = [0.0, 0.01, 0.04, 0.16]
        corr_spatial = corr_temporal
        graphs = ["sparse", "complete"]

    if experiment == "t-vs-f_time-space":
        # Compare T vs F
        F_list = [  1, 16, 64, 256, 1024,  1,  1,   1,    1]
        T_list = [  1,  1,  1,   1,    1, 16, 64, 256, 1024]
        corr_temporal = [0.0, 0.01, 0.04, 0.16]
        corr_spatial = corr_temporal
        distrib = ["norm"]
        graphs = ["sparse"]

    if experiment == "t-vs-f_no-space":
        # Compare T vs F
        F_list = [  1, 16, 64, 256, 1024,  1,  1,   1,    1]
        T_list = [  1,  1,  1,   1,    1, 16, 64, 256, 1024]
        corr_temporal, corr_spatial = to_grid([0.0, 0.01, 0.04, 0.16], [0])
        distrib = ["norm"]
        graphs = ["sparse"]

    if experiment == "debug-full0":
        rep = 1000
        F_list, T_list = to_grid([1], [1])
        # corr_spatial = [0.0, 0.01, 0.04, 0.16]
        # corr_temporal, corr_spatial = to_grid([0.0, 0.01, 0.04, 0.16], [0])
        corr_temporal, corr_spatial = to_grid([0], [0.0, 0.01, 0.04, 0.16])
        graphs = ["complete"]

    common_args = dict(F_list=F_list, T_list=T_list,
                       distrib_list=distrib, cs_list=corr_spatial, ct_list=corr_temporal,
                       repetitions=rep, alpha=alpha, disable_warning=disable_warning)

    from triangular_tricom_graph import Graph
    edge_index_complete = [[i, j] for i in range(G.num_nodes) for j in range(G.num_nodes) if i != j]
    edge_index_complete = np.array(edge_index_complete).T
    G_full = Graph(edge_index=edge_index_complete, node_position=G.node_position)

    for graph in graphs:
        if graph == "sparse":
            considered_edge_index = G.edge_index
        elif graph == "complete":
            considered_edge_index = G_full.edge_index
        else:
            raise NotImplementedError()

        df, x = run_multiple_simulations(G=G, considered_edge_index=considered_edge_index, **common_args)
        df.to_csv(path_or_buf=RESULT_FOLDER + "/dataframe_" + experiment + ".csv")
        name = experiment if graph == "sparse" else experiment+"_"+graph
        print(f"Plotting experiment {name}")
        plot_results(df=df, name=name)
        make_table(df=df)
        if rep == 1:
            G.plot()

if __name__ == "__main__":

    #Figure 3
    main("power-unimodal", disable_warning=True)
    main("power-mixture", disable_warning=True)
    #Figure 5 (Supp. Mat.)
    main("viz")
    #Figure 7 (Supp. Mat.)
    main("sparse-full")

