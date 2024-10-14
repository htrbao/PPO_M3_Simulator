import streamlit as st
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from test.level_generate import get_real_levels
matplotlib.use('Agg')
st.set_page_config(page_title="M3 Testing Metrics", layout="wide", initial_sidebar_state='collapsed')

if 'level' not in st.session_state:
    st.session_state['level'] = pd.DataFrame(get_real_levels(full_info=True))

store_dir = "_saved_test"

# Function to load all CSVs from a given folder
def load_csv_files(store_dir, dir_list):
    csv_files = []
    for folder in dir_list: 
        for f in os.listdir(os.path.join(store_dir, folder)):
            if f.endswith('.csv'):
                csv_files.append({
                    "path": os.path.join(store_dir, folder, f),
                    "name": folder
                        })
    print(csv_files)
    dataframes = {}
    for model in csv_files:
        dataframes[model['name']] = pd.read_csv(model['path'])
    return dataframes

def draw_line(df, X, Y, title, ax):
    x_axis=np.arange(len(X))
    ax.set_xticks(x_axis, x_axis)
    ax.set_ylim([0, 100])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title(title, fontweight="bold")
    
    def _draw_line(y_axis, label):
        ax.plot(x_axis, y_axis, label=label)
        for x ,y in zip(x_axis, y_axis):
            ax.annotate(str(y)+'%',xy=(x,y), color='black', ha='center', va='bottom',xytext=(0, 1),  # 4 points vertical offset.
                            textcoords='offset points', fontsize=6)
        
    for model, sub_df in df.items():
        y_axis = (sub_df.groupby(by=['realm_id'])[Y].mean() * 100).round(1).tolist()
        _draw_line(y_axis, model)



    
def draw_plot(df):
    realms = list(df.values())[0]['realm_id'].unique().tolist()
    with st.container():
        ratio = (14,8)
        fig, axes  = plt.subplots(2, 2, figsize=ratio, sharey=True, )
        
        draw_line(df, X=realms, Y='win_rate', title="Win rate", ax = axes[0, 0])

        draw_line(df, X=realms, Y='hit_rate', title="Hit rate", ax = axes[0, 1])
        fig.subplots_adjust(hspace=0.2, wspace =0.01)
        fig.legend(list(df.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.01),
            fancybox=True, shadow=True, ncol=1)
        # fig.set_dpi(300)
        # st.pyplot(fig, use_container_width=False)
        
        # fig, axes  = plt.subplots(1, 2, figsize=ratio, sharey=True)
        draw_line(df, X=realms, Y='remain_hp_monster', title="Remaining HP of Monster", ax = axes[1, 0])

        draw_line(df, X=realms, Y='avg_damage_per_hit', title="AVG Damage per Hit", ax = axes[1, 1])
        # fig.subplots_adjust(hspace=0.1, wspace =0.1)
        # fig.legend(list(df.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.01),
        #     fancybox=True, shadow=True, ncol=1)
        with st.columns([1, 10, 1])[1]:
            st.pyplot(fig, use_container_width=False)

def draw_common_stat(df):
    avg_win_rate = df['win_rate'].mean()
    avg_hit_rate = df['hit_rate'].mean()
    avg_damage_per_hit = df['avg_damage_per_hit'].mean()
    remain_hp_monster = df[df['remain_hp_monster'] > 0]["remain_hp_monster"].mean()
    
    col2, col3, col4, col5 = st.columns(4)
    
    col2.metric("Average Win Rate", f"{avg_win_rate*100:.2f}%")
    col3.metric("Average Hit Rate", f"{avg_hit_rate*100:.2f}%")
    col4.metric("Average Remaining HP of Monster (Lose Only)", f"{remain_hp_monster*100:.2f}%")
    col5.metric("Average Damage per Hit", f"{avg_damage_per_hit*100:.2f}%")


# Main Streamlit app
st.title("Comparing M3 model")

# Input folder name
dir_list = [d for d in os.listdir(store_dir) if os.path.isdir(os.path.join(store_dir, d))]



dfs = load_csv_files(store_dir, dir_list)
dfs = dict(sorted(dfs.items()))
st.success(f"Loaded {len(dfs)} model.")
st.header("Higest win rate model statistics", divider=True)

highest_model = list(dict(sorted(dfs.items(), key=lambda x: x[1]['win_rate'].mean(), reverse=True)).keys())[0]
df = dfs[highest_model]
df = df.sort_values(by=['realm_id', 'node_id'])
st.subheader(f'Highest win rate model: {highest_model}')
draw_common_stat(df)
st.header("Statistics", divider=True)
draw_plot(dfs)
st.header("Highest model: ", divider=True)
st.dataframe(df, use_container_width=True)
    

