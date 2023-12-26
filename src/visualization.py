# STANDARD LIBRARIES
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../src"))
import jax.numpy as np
import skbio.stats.composition as cmp
import toyplot
import toyplot.pdf, toyplot.png
# plotting functionalities
from plot_fct import plot_mse_results, plot_beta_results
import ibis_util_functions as util

def plot_results(df_beta, df_mse, betaT, filter_list, sort_to_filter=True):
    """
    plot results
    """
    p = len(df_beta.iloc[0, 1])  # get number of microbes
    V = cmp._gram_schmidt_basis(p)

    # MSE Plot
    fig1 = plot_mse_results(df_mse, filter_list, sort_to_filter=sort_to_filter)
    fig1.show()

    # Influential Beta Plot
    fig2 = plot_beta_results(df_beta, V.T @ betaT, filter_list, sort_to_filter=sort_to_filter)
    fig2.update_layout(showlegend=False)
    fig2.show()

    # Non-influential Beta Plot
    fig3 = plot_beta_results(df_beta, V.T @ betaT, filter_list, sort_to_filter=sort_to_filter, beta_zero=True)
    fig3.update_layout(showlegend=False)
    fig3.show()

    return fig1, fig2, fig3


def write_result_table(df_beta, df_mse, betaT):
    """
    write results for result table
    """

    p = len(df_beta.iloc[0, 1])  # get number of microbes
    V = cmp._gram_schmidt_basis(p)

    def mean(x): return np.around(x.mean(), 2)

    def std_err(x): return np.around(x.std() / np.sqrt(len(x)), 2)

    def beta_error(x):
        return ((x - V.T @ betaT) ** 2).sum()

    # Support Recovery
    eps = 10e-6

    def support_recovery_0(x):
        return ((x < eps) & (x > -eps) & ((V.T @ betaT) < eps) & ((V.T @ betaT) > -eps)).sum() / (
                    (V.T @ betaT < eps) & (V.T @ betaT > -eps)).sum()

    df_beta["Zero Recovery"] = df_beta["Beta"].apply(support_recovery_0)

    def false_non_zero_val(x):
        return (((x < -eps) | (x > eps)) & ((V.T @ betaT < eps) & (V.T @ betaT > -eps))).sum()

    def false_zero_val(x):
        return (((x < eps) & (x > -eps)) & ((V.T @ betaT > eps) | (V.T @ betaT < -eps))).sum()


    # Prediction Error
    df_prediction_error = df_mse.groupby("Method").agg({"MSE": [mean, std_err]})

    # Estimation Error
    df_beta["Estimation Error"] = df_beta["Beta"].apply(beta_error)
    df_beta["Estimation Error"] = df_beta["Estimation Error"].apply(lambda x: float(x))
    df_estimation_error = df_beta.groupby("Method").agg({"Estimation Error": [mean, std_err]})

    # Zero Val
    df_beta["False NON Zero Values"] = df_beta["Beta"].apply(false_non_zero_val)
    df_beta["False NON Zero Values"] = df_beta["False NON Zero Values"].apply(lambda x: float(x))
    df_support_zeroval = df_beta.groupby("Method").agg({"False NON Zero Values": [mean, std_err]})

    # Non Zero Val
    df_beta["False Zero Values"] = df_beta["Beta"].apply(false_zero_val)
    df_beta["False Zero Values"] = df_beta["False Zero Values"].apply(lambda x: float(x))
    df_support_nonzeroval = df_beta.groupby("Method").agg({"False Zero Values": [mean, std_err]})

    res = {
        "df_prediction_error": df_prediction_error,
        "df_estimation_error": df_estimation_error,
        "df_support_zeroval": df_support_zeroval,
        "df_support_nonzeroval": df_support_nonzeroval
    }
    return res


def draw_tree(node_df, title="", agg_level="Class", tip_labels=True, save_path=None, eff_max=None):
    """
    Plots a nice tree from a tascCODA node_df

    :param node_df: DataFrame
        Must have columns ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Final Parameter"];
        index must be full taxonomic names, e.g. "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae*Alistipes"
    :param title: string
        Title that is printed at the top of the plot
    :param add_sccoda: bool
        Write genera found to be DA by scCODA in red
    :param tip_labels: bool
        If true, add genus names as tip labels
    :param save_path: string
        If not none, plot will be saved as svg to this file
    :param eff_max: float
        Maximum absolute value of effects for scaling of node size.
        If None, ,aximum absolute value of node_df["Final Parameter"] will be used
    :return:
    """

    # Build dictionary tu use with util.build_fancy_tree
    node_df = node_df.reset_index()

    tree_dat = {}

    tree_dat[agg_level] = node_df  # node_df[node_df[agg_level]!=""]

    # Make toytree object, initialize toyplot canvas
    tree, markers = util.build_fancy_tree(tree_dat, agg_level)

    canvas = toyplot.Canvas(width=1000, height=1000)

    # Area for tree plot
    # ax0 = canvas.cartesian(bounds=(50, 800, 50, 800), padding=0)
    ax0 = canvas.cartesian(bounds=(0, 1000, 0, 1000), padding=0)
    ax0.x.show = False
    ax0.y.show = False

    # Determine max effect size, if necessary
    if eff_max is None:
        eff_max = np.max([np.abs(n.effect) for n in tree.treenode.traverse()])

    # label_colors = [node_df.loc[node_df[agg_level]==n, "Final Parameter"] for n in tree.get_tip_labels()]

    label_colors = ["lightgray" if len(node_df.loc[node_df[agg_level] == n, "Final Parameter"].to_list()) != 1
                    else str(node_df.loc[node_df[agg_level] == n, "Final Parameter"].item())
                    for n in tree.get_tip_labels()]

    # label_colors = [str(node_df.loc[node_df[agg_level]==n, "Final Parameter"].item())
    #                for n in tree.get_tip_labels()]
    # draw tree.

    tree.draw(
        axes=ax0,
        layout='c',  # circular layout
        edge_type='p',  # rectangular edges
        node_sizes=[(np.abs(n.effect) * 20 / eff_max) + 10 if n.effect != 0 else 0 for n in tree.treenode.traverse()],
        # node size scales with node feature "effect"
        node_colors=[n.color for n in tree.treenode.traverse()],  # node color from node feature "color"
        node_style={
            "stroke": "black",
            "stroke-width": "1",
        },
        width=800,
        height=800,
        tip_labels=tip_labels,  # Print tip labels or not
        tip_labels_align=True,
        tip_labels_style={"font-size": "20px"},
        tip_labels_colors=label_colors,
        edge_colors=[tree.idx_dict[x[1]].edge_color for x in tree.get_edges()],
        # edge color from node feature "edge_color"
        edge_widths=3  # width of tree edges
    )

    # add area for plot title
    ax1 = canvas.cartesian(bounds=(50, 800, 50, 100), padding=0, label=title)
    ax1.x.show = False
    ax1.y.show = False

    # add legend for phylum colors
    canvas.legend(markers,
                  bounds=(0, 100, 50, 200),
                  label="Phylum"
                  )

    markers2 = [
        (f"Only Second LC", toyplot.marker.create(shape="o", size=20, mstyle={"fill": "blue"})),
        (f"ILR+LC", toyplot.marker.create(shape="o", size=20, mstyle={"fill": "orange"})),
        (f"Both Methods", toyplot.marker.create(shape="o", size=20, mstyle={"fill": "black"})),
        # (f"DIR+LC",toyplot.marker.create(shape="o", size=20, mstyle={"fill": "red"}))
    ]
    canvas.legend(markers2,
                  bounds=(800, 1000, 50, 200),
                  label=f"Influential Compositions"
                  )

    # save plot if desired

    if save_path is not None:
        toyplot.svg.render(canvas, save_path)

    return tree