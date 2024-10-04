        
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import matplotlib
import numpy as np
import pandas as pd
import json
import streamlit as st
import json

from test.level_generate import get_real_levels
matplotlib.use('Agg')

st.set_page_config(page_title="M3 Testing Metrics", layout="wide", initial_sidebar_state='collapsed')

if 'level' not in st.session_state:
    st.session_state['level'] = pd.DataFrame(get_real_levels(full_info=True))

ID2MONS = {
    '15': "Block Monster",
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

    ax.set_xticks(np.arange(data.shape[1])-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0])-.5, minor=True)
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

def draw_heatmap(data_map, num_level, type, legend=None, **kwargs):
    np_map = np.array(data_map) * -1 if type == "wall" else np.array(data_map)
    if type == 'map':
        np_map = np.where(np_map == -1, 1, np_map)
    height, width = np_map.shape

    fig, ax = plt.subplots()

    im, cbar = heatmap(np_map, np.arange(height), np.arange(width), ax=ax,
                     vmin=0, **kwargs)
    if type != 'map':
        texts = annotate_heatmap(im, valfmt='{x}')

    if type == 'map':
        ax.legend(handles=legend, title="Legend", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{type} {num_level}", fontweight="bold")
    else:
        ax.set_title(f"{type} rate ({num_level} levels)", fontweight="bold")
    return fig 

st.subheader("Show map")
with st.container():
    with st.form('show_map'):
        col1, col2 = st.columns(2)
        with col1:
            realm_id = st.number_input("Realm ID", min_value=0, max_value=5)
        with col2:
            node_id = st.number_input("Node ID", min_value=0, max_value=100)
            
        submitted = st.form_submit_button("Submit")
        if submitted:
            df = st.session_state['level']
            map = df[(df['realm_id'] == realm_id) & (df['node_id'] == node_id)]
            map = map.reset_index()
            level = map.at[0, 'level']
            monsters = map.at[0, 'monsters']
            max_step = map.at[0,'max_step']
            colors = [(0.7, 0.7, 0.7), (0, 0, 0), (1, 0, 0), (1, 0, 0)]  # White, Black, Red
            values = [0, 1/15, 14/15, 1]  # Mapping values (0 for white, 1 for black, 15 or more for red)
            custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(values, colors)))
            legend_elements = [
                Patch(facecolor=colors[0], edgecolor='black', label='Tile'),
                Patch(facecolor=colors[1], edgecolor='black', label='Wall'),
                Patch(facecolor=colors[2], edgecolor='black', label='Monster')
            ]
            fig = draw_heatmap(level.board,  f"realm {realm_id}, node {node_id}", "map", legend=legend_elements, cmap=custom_cmap)
            with st.columns([1, 3, 1])[1]:
                st.pyplot(fig, use_container_width=False)
            st.subheader(f"Max step: {max_step}")
            st.subheader(f"Num tile: {level.n_shapes}")
            st.subheader(f"Monsters:")
            st.write(monsters)
