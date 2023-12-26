import os
import pandas as pd
import numpy as np
import toytree as tt
import toyplot


tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]





def get_phylo_levels(results, level, col="Cell Type"):

    """
    Get a taxonomy table (columns are "Kingdom", "Phylum", ...) from a DataFrame where the one column contains full taxon names.

    :param results: pandas DataFrame
        One column must be strings of the form "<Kingdom>*<Phylum>*...", e.g. "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae*Alistipes"
    :param level: string
        Lowest taxonomic level (from ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]) that should be included
    :param col: string
        Name of the column with full taxon names
    :return:
    DataFrame with columns ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]

    """

    max_level_id = tax_levels.index(level)+1
    cols = tax_levels

    tax_table = pd.DataFrame(columns=cols, index=np.arange(len(results)))
    for i in range(len(results)):
        char = results.loc[i, col]
        split = char.split(sep="*")
        for j in range(max_level_id):
            try:
                tax_table.iloc[i, j] = split[j]
            except IndexError:
                tax_table.iloc[i, j] = None

    return tax_table


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    :param df_:
    :param a:
    :param i:
    :param innerl:
    :return:
    """
    if i+1 < df_.shape[1]:
        a_inner = pd.unique(df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i+1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i+1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(df, agg_level, inner_label=True):
    """
    Converts a taxonomy DataFrame into a Newick string
    :param df: DataFrame
        Must have columns ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    :param inner_label: Boolean
        If True, internal nodes in the tree will keep their respective names
    :return:
        Newick string
    """
    if agg_level == "Phylum":
        df = df.drop(columns=["Class", "Order", "Family", "Genus"])
    if agg_level == "Class":
        df = df.drop(columns=["Order", "Family", "Genus"])
    if agg_level == "Order":
        df = df.drop(columns=["Family", "Genus"])
    if agg_level == "Family":
       df = df.drop(columns=["Genus"])

    tax_levels = [col for col in df.columns if col in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]]
    df_tax = df.loc[:, tax_levels]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def build_fancy_tree(data, agg_level):
    """
    Make a toytree object with all kinds of extras, like edge colors, effect sizes, ...

    :param data: Dictionary of DataFrames.
        Contains effect sizes, taxonomy info, etc. for all taxonomic ranks
        Dictionary keys should be ["Phylum", "Class", "Order", "Family", "Genus"]
        Each DataFrame should have the columns "Final Parameter" (= effect size), as well as columns for all taxonomic ranks
    :return:
        toytree object
    """

    # Get genus-level data for building a complete tree
    data_gen = data[agg_level]
    # Combine results from all levels into one df
    data_all = pd.concat(data.values())

    # Get newick string and build toytree with no extras
    newick = df2newick(data_gen, agg_level)
    #print(newick)
    tree = tt.tree(newick, tree_format=1)

    # Edge colors.
    # The color dictionary is hard-coded, keys must be the names of all phyla in the data
    palette = toyplot.color.brewer.palette("Set1") + toyplot.color.brewer.palette("Set3")
    edge_color_dict = {
        "Firmicutes": palette[11],
        "Proteobacteria": palette[14],
        "Bacteroidota": palette[2],
        "Actinobacteria": palette[3],
        "Verrucomicrobia": palette[4],
        "Cyanobacteria": palette[15],
        "WPS-2": palette[6],
        "TM7": palette[7],
        "Tenericutes": palette[8],
        "Bacteroidetes": palette[9]
    }

    # marker objects for plotting a legend
    markers = []

    # Height of the colored level (here: 2nd highest)
    max_height = np.max([n.height - 2 for n in tree.treenode.traverse()])

    # Iterate over all tree nodes and assign edge colors
    c = 0
    for n in tree.treenode.traverse():
        # If node corresponds to the colored level (here: Phylum), determine color and assign it as feature "edge_color" to all descendants
        if n.height == max_height:
            col = edge_color_dict[n.name]
            n.add_feature("edge_color", col)
            for n_ in n.get_descendants():
                n_.add_feature("edge_color", col)

            # Also add a marker for the legend
            col2 = '#%02x%02x%02x' % tuple([int(255*x) for x in col.tolist()[:-1]])
            m = toyplot.marker.create(shape="o", size=8, mstyle={"fill": col2})
            markers.append((n.name, m))

            c += 1
        # For all levels above the colored level, assign edge color black
        elif n.height > max_height:
            n.add_feature("edge_color", "black")

    # assign taxonomic levels to nodes (e.g. "Genus" for a node on the lowest level)
    for n in tree.treenode.traverse():
        if n.height == "":
            l = tax_levels[-1]
        elif n.height >= len(tax_levels):
            l = ""
        else:
            l = tax_levels[-(int(n.height) + 1)]
        n.add_feature("tax_level", l)

    # add effects to each node as feature "effect":
    # For all results, add the taxonomic rank (forgot to do that when combining initially)
    data_all["level"] = "Kingdom"
    for l in tax_levels[1:]:
        data_all.loc[pd.notna(data_all[l]), "level"] = l

    # Iterate over all tree nodes
    for n in tree.treenode.traverse():
        # Catch root node
        if n.tax_level == "":
            n.add_feature("effect", 0)
        else:
            # Find row in the effect DataFrame that mathches in taxonomic rank and name
            l = data_all.loc[(data_all["level"] == n.tax_level) & (data_all[n.tax_level] == n.name), :]
            # If there is only one matching row, assign effect size as node feature
            if len(l) == 1:
               #n.add_feature("effect", l["Final Parameter"].values[0])
                n.add_feature("effect", 1)
                n.add_feature("color", l["Final Parameter"].values[0])
            # If there is no corresponding row, assign effect size 0
            elif len(l) == 0:
                n.add_feature("effect", 0)
                n.add_feature("color", "cyan")
            # If there are multiple corresponding rows (might happen if e.g. genera from two different families have the same name),
            # solve by comparing entire taxonomy assignment and assign effect size
            elif len(l) > 1:
                par_names = [n.name] + [m.name for m in n.get_ancestors()][:-1]
                par_names.reverse()
                full_name = '*'.join(par_names)
                ll = l[l["Cell Type"] == full_name]
                n.add_feature("effect", ll["Final Parameter"].values[0])
                n.add_feature("color", ll["Final Parameter"].values[0])
                n.add_feature("effect", 1)

                #n.add_feature("effect", l["Final Parameter"].values[0])


    # add node colors as feature "color", depending on whether effects are positive (black) or negative (white).
    # Effects of size 0 have no impact, as their markers will have size 0, but are colored in cyan for completeness.
    for n in tree.treenode.traverse():
        #print(n.effect)
        if np.sign(n.effect) == 1:
            n.add_feature("color", "black")
        elif np.sign(n.effect) == -1:
            n.add_feature("color", "white")
        else:
            n.add_feature("color", "cyan")

    return tree, markers
