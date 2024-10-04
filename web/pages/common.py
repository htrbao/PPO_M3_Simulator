import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
import matplotlib
import numpy as np
import pandas as pd
import json
import streamlit as st

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

def draw_heatmap(data_map, num_level, type, legend=None, **kwargs):
    np_map = np.array(data_map) * -1 if type == "wall" else np.array(data_map)
    if type == 'map':
        np_map = np.where(np_map == -1, 1, np_map)
    height, width = np_map.shape

    fig, ax = plt.subplots()

    im, cbar = heatmap(np_map, np.arange(height)+1, np.arange(width)+1, ax=ax,
                     vmin=0, **kwargs)
    if type != 'map':
        texts = annotate_heatmap(im, valfmt='{x}')

    if type == 'map':
        ax.legend(handles=legend, title="Legend", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{type} {num_level}", fontweight="bold")
    else:
        ax.set_title(f"{type} rate ({num_level} levels)", fontweight="bold")
    return fig 

with st.container():
    st.header("Monster Statistic", divider=True)
    realms = []
    monster_data = {v: [] for v in ID2MONS.values()}
    monster_data_total = {v: 0 for v in ID2MONS.values()}
    total_monsters = 0
    avg_monster_per_realm = []
    for realm, data in statistic_data.items():
        realms.append(f"realm {realm}")
        total_monster = sum(data["monsters"].values())
        total_monsters += total_monster
        avg_monster_per_realm.append(total_monster / data["num_levels"])
        for k, v in data["monsters"].items():
            monster_data[ID2MONS[k]].append(v*100/total_monster)
            monster_data_total[ID2MONS[k]] += v
    
    
    realms.append("total")
    avg_monster_per_realm.append(total_monsters / len(st.session_state['level']))
    for k, v in monster_data_total.items():
        monster_data[k].append(v*100/total_monsters)
    width = 0.5
    fig, ax = plt.subplots(figsize=(5, 4))
    ax2 = ax.twinx()
    bottom = np.zeros(len(realms))
    
    for monster_type, monster_num in monster_data.items():
        p = ax.bar(realms, monster_num, width, label=monster_type, bottom=bottom)
        bottom += monster_num

    ax.set_ylabel('Weight of Monsters')
    ax.set_title('Weight of Monsters per Realm')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    ax2.set_ylim(1, 2.5)
    ax2.set_ylabel('AVG Monster per Level')
    
    ax2.plot(ax.get_xticks(), avg_monster_per_realm, color="black")
    with st.columns([1,4,1])[1]:
        st.pyplot(fig, use_container_width=False)


def draw_bar(df, realm):
    fig, ax = plt.subplots(figsize=(20, 7))
    sub_df = df[df['realm_id'] == realm]
    
    colors = ['g' if m >= 0 else 'r' for m in sub_df['max_step_&_monster_hp'].tolist()]
    ax = sub_df.plot.bar(x='node_id', y='max_step_&_monster_hp', ax=ax, rot=65, title=f"Realm {realm}", color=colors,edgecolor='black')
    ax.get_legend().remove()
    for bar in ax.patches:
        if bar.get_height() < 0:
            ax.annotate(str(format(bar.get_height(), '.0f')), 
                        (bar.get_x() + bar.get_width() / 2, 
                            bar.get_height()), ha='center', va='center',
                        xytext=(0, -8),
                        textcoords='offset points')
        else:
            ax.annotate(str(format(bar.get_height(), '.0f')), 
                        (bar.get_x() + bar.get_width() / 2, 
                            bar.get_height()), ha='center', va='center',
                        xytext=(0, 8),
                        textcoords='offset points')
    ax.annotate("Higher is easier to win the level.",
            xy = (0.5, -0.15),
            xycoords='axes fraction',
            ha='center',
            va="center",
            fontsize=15)
    return fig


with st.container():
    df = st.session_state['level']
    realms = df['realm_id'].unique().tolist()
    df['max_step_&_monster_hp'] = df.apply(lambda row: row['max_step'] - sum([mon['kwargs']['hp'] for mon in row['monsters']]), axis=1)
    st.header("Difference between Max Step and Monster HP", divider=True)
    
    for realm in realms:
        with st.columns([1,8, 1])[1]:
            if f'common_realm_{realm}' not in st.session_state:
                fig = draw_bar(df, realm)
                st.session_state[f'common_realm_{realm}'] = fig
            else:
                fig = st.session_state[f'common_realm_{realm}']
            st.pyplot(fig, use_container_width=False)
    pass

with st.container():

    st.header(f"Map Statistic", divider=True) 

    for id, (realm, data) in enumerate(statistic_data.items()):
        if id % 2 == 0:
            col = st.columns(2)
        with col[id % 2]:
            st.subheader("Realm: " + realm, divider="rainbow")
            col1, col2 = st.columns(2)
            with col1:
                if f"realm_{realm}_wall" not in st.session_state:
                    fig = draw_heatmap(data["wall_map"], data["num_levels"], "wall", cmap="BuGn")
                    st.session_state[f'realm_{realm}_wall'] = fig
                else:
                    fig = st.session_state[f'realm_{realm}_wall']
                st.pyplot(fig)
            with col2:
                if f"realm_{realm}_monster" not in st.session_state:
                    fig = draw_heatmap(data["monster_map"], data["num_levels"], "monster", cmap="BuGn")
                    st.session_state[f'realm_{realm}_monster'] = fig
                else:
                    fig = st.session_state[f'realm_{realm}_monster']
                st.pyplot(fig)

# print(st.session_state['level'])
    
