import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import json
import streamlit as st

st.set_page_config(page_title="M3 Testing Metrics", layout="wide")


ID2MONS = {
    '15': "Damage Block Monster",
    '2': "Have Paper Monster",
    '17': "Throw Blocker",
    "default": "Damage Monster" 
}

store_dir = "_saved_test/statistic_realm"
with open('test/config/statistic_realm.json', 'r') as f:
    statistic_data = json.load(f)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbarlabel:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
            rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def draw_heatmap(data_map, num_level, type, **kwargs):
    np_map = np.array(data_map) * -1 if type == "wall" else np.array(data_map)
    # np_map = np_map/ num_level
    # np_map = np_map.round(2)
    height, width = np_map.shape

    fig, ax = plt.subplots()

    im, cbar = heatmap(np_map, np.arange(height)+1, np.arange(width)+1, ax=ax,
                     vmin=0, **kwargs)
    texts = annotate_heatmap(im, valfmt="{x}")
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_title(f"{type} rate ({num_level} levels)", fontweight="bold")
    return fig 

with st.container():
    st.header("Monster Statistic", divider=True)
    realms = []
    monster_data = {v: [] for v in ID2MONS.values()}
    avg_monster_per_realm = []
    for realm, data in statistic_data.items():
        realms.append(f"realm {realm}")
        avg_monster_per_realm.append(sum(data["monsters"].values()) / data["num_levels"])
        for k, v in data["monsters"].items():
            monster_data[ID2MONS[k]].append(v)
            
    width = 0.5
    fig, ax = plt.subplots(figsize=(5, 4))
    ax2 = ax.twinx()
    bottom = np.zeros(len(realms))
    
    for monster_type, monster_num in monster_data.items():
        p = ax.bar(realms, monster_num, width, label=monster_type, bottom=bottom)
        bottom += monster_num

    ax.set_ylabel('Number of Monsters')
    ax.set_title('Number of Monsters per Realm')
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 150)
    
    print(avg_monster_per_realm)
    ax2.set_ylim(1, 2.5)
    ax2.set_ylabel('AVG Monster per Level')
    
    ax2.plot(ax.get_xticks(), avg_monster_per_realm, color="black")
    with st.columns([1,4,1])[1]:
        st.pyplot(fig, use_container_width=False)
    

st.header(f"Map Statistic", divider=True) 

for id, (realm, data) in enumerate(statistic_data.items()):
    if id % 2 == 0:
        col = st.columns(2)
    with col[id % 2]:
        st.subheader("Realm: " + realm, divider="rainbow")
        col1, col2 = st.columns(2)
        with col1:
            fig = draw_heatmap(data["wall_map"], data["num_levels"], "wall", cmap="YlGn")
            st.pyplot(fig)
        with col2:
            fig = draw_heatmap(data["monster_map"],  data["num_levels"], "monster", cmap="magma_r")
            st.pyplot(fig)
    