import streamlit as st
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib import ticker
from test.level_generate import get_real_levels
matplotlib.use('Agg')
st.set_page_config(page_title="M3 Testing Metrics", layout="wide", initial_sidebar_state='collapsed')

if 'level' not in st.session_state:
    st.session_state['level'] = pd.DataFrame(get_real_levels(full_info=True))
    
if 'old' not in st.session_state:
    old = {
        "num_level": 0,
        "avg_win_rate": 0,
        "avg_hit_rate": 0,
        "avg_damage_per_hit": 0,
        "remain_hp_monster": 0,
    }
    st.session_state['old'] = old

store_dir = "_saved_test"

# Function to load all CSVs from a given folder
def load_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df.info()
        return df

def draw_line(x_axis, y_axis, color, label):
    fig, ax  = plt.subplots(figsize=(5, 5))
    
    ax.plot(x_axis, y_axis, label=label, color=color)
    ax.set_xticks(x_axis, [f"realm {realm}" for realm in x_axis])
    ax.legend(loc='best', ncols=1)
    ax.set_ylim([0, 100])
    
    for x ,y in zip(x_axis, y_axis):
        ax.annotate(str(y)+'%',xy=(x,y), color='black', ha='center', va='bottom',xytext=(0, 1),  # 4 points vertical offset.
                        textcoords='offset points')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    return fig

def draw_bar(df, realm):
    fig, ax = plt.subplots(4, 1, figsize=(20, 9), sharex=True)
    sub_df = df[df['realm_id'] == realm]
    sub_df.loc[:, "win_rate"] = sub_df["win_rate"] * 100
    sub_df.loc[:, "hit_rate"] = sub_df["hit_rate"] * 100
    sub_df.loc[:, "remain_hp_monster"] = sub_df["remain_hp_monster"] * 100
    # sub_df.plot.bar(x='node_id', y='win_rate', figsize=(10, 5), title=f"Realm {realm} Win Rate", ax=ax, rot=60)
    axes = sub_df.plot.bar(x='node_id', y=['win_rate', 'hit_rate', 'remain_hp_monster'], subplots=True, ax=ax[0: 3], rot=65, title=f"Realm {realm}")
    for axs in axes:
        axs.yaxis.set_major_formatter(ticker.PercentFormatter())
        axs.set_ylim([0, 100])
        for bar in axs.patches:
            if bar.get_height() < 100 and bar.get_height() > 0:
                axs.annotate(str(format(bar.get_height(), '.0f')) + '%', 
                            (bar.get_x() + bar.get_width() / 2, 
                                bar.get_height()), ha='center', va='center',
                            xytext=(0, 8),
                            textcoords='offset points')
    draw_bar_monster(sub_df, realm, ax[3])
    return fig


def draw_bar_monster(df, realm, ax):
    
    colors = ['g' if m >= 0 else 'r' for m in df['max_step_&_monster_hp'].tolist()]
    ax = df.plot.bar(x='node_id', y='max_step_&_monster_hp', ax=ax, rot=65, title=f"Difference between Max Step and Monster HP", color=colors)
    pa1 = Patch(facecolor='red')
    pa2 = Patch(facecolor='green')
    ax.legend(handles=[pa1, pa2], labels=["", "difference_max_step_&_monster_hp"],loc='best', ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,)

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
            xy = (0.5, -0.4),
            xycoords='axes fraction',
            ha='center',
            va="center",
            fontsize=13)



# with st.container():
    # realms = df['realm_id'].unique().tolist()
    # st.header("", divider=True)

    
def draw_plot(df):
    realms = df['realm_id'].unique().tolist()

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['win_rate'].mean() * 100).round(1).tolist(), color="blue", label="Win rate")
        with col1:
            st.pyplot(fig)
        
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['hit_rate'].mean() * 100).round(1).tolist(), color='red', label="Hit rate")
        with col2:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['remain_hp_monster'].mean() * 100).round(1).tolist(), color='teal', label="Remaining HP of Monster (Lose Only)")
        with col3:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['avg_damage_per_hit'].mean() * 100).round(1).tolist(), color='teal', label="AVG Damage per Hit")
        with col4:
            st.pyplot(fig)
    st.header("Realm Win Rate")
    
    level_df = st.session_state['level']
    level_df['max_step_&_monster_hp'] = level_df.apply(lambda row: row['max_step'] - sum([mon['kwargs']['hp'] for mon in row['monsters']]), axis=1)
    df = pd.merge(df, level_df, on=['realm_id', 'node_id'], how="inner")
    for realm in realms:
        with st.columns([1,8, 1])[1]:
            if f'realm_{realm}' not in st.session_state:
                fig = draw_bar(df, realm)
                st.session_state[f'realm_{realm}'] = fig
            else:
                fig = st.session_state[f'realm_{realm}']
            st.pyplot(fig, use_container_width=False)


def draw_common_stat(df):
    
    level = len(df)
    num_games = df['num_games'].max()
    
    avg_win_rate = df['win_rate'].mean()
    avg_hit_rate = df['hit_rate'].mean()
    avg_damage_per_hit = df['avg_damage_per_hit'].mean()
    remain_hp_monster = df[df['remain_hp_monster'] > 0]["remain_hp_monster"].mean()
    st.subheader(f"Test on {num_games} games per levels ({level} levels)")
    
    col2, col3, col4, col5 = st.columns(4)
    
    
    col2.metric("Average Win Rate", f"{avg_win_rate*100:.2f}%", delta=f"{(avg_win_rate - st.session_state['old']['avg_win_rate'])*100:.2f}%")
    col3.metric("Average Hit Rate", f"{avg_hit_rate*100:.2f}%", delta=f"{(avg_hit_rate - st.session_state['old']['avg_hit_rate'])*100:.2f}%")
    col4.metric("Average Remaining HP of Monster (Lose Only)", f"{remain_hp_monster*100:.2f}%", delta=f"{(remain_hp_monster - st.session_state['old']['remain_hp_monster'])*100:.2f}%", delta_color ="inverse")
    col5.metric("Average Damage per Hit", f"{avg_damage_per_hit*100:.2f}%", delta=f"{(avg_damage_per_hit - st.session_state['old']['avg_damage_per_hit'])*100:.2f}%")

    st.session_state['old']["avg_win_rate"] = avg_win_rate
    st.session_state['old']["avg_hit_rate"] = avg_hit_rate
    st.session_state['old']["avg_damage_per_hit"] = avg_damage_per_hit
    st.session_state['old']["remain_hp_monster"] = remain_hp_monster


# Main Streamlit app
st.title("CSV File Loader and Plotter")

# Input folder name
dir_list = [d for d in os.listdir(store_dir) if os.path.isdir(os.path.join(store_dir, d))]

folder_path = st.selectbox("Select the model:", options=dir_list)

# Load CSVs when folder path is provided
if folder_path:

    df = load_csv_files(os.path.join(store_dir, folder_path))
    st.success(f"Loaded {len(df)} rows.")
    st.header("Common statistics", divider=True)
    draw_common_stat(df)
    draw_plot(df)
    st.dataframe(df, use_container_width=True)
