import streamlit as st
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker
from gym_match3.envs.game import BoxMonster

import numpy as np
from test.level_generate import get_real_levels
matplotlib.use('Agg')
st.set_page_config(page_title="M3 Testing Metrics", layout="wide", initial_sidebar_state='collapsed')


def identify_monster_type(monster_data):
    if "request_masked" in monster_data["kwargs"]:
        return "Block Monster"
    elif "have_paper_box" in monster_data["kwargs"]:
        return "Have Paper Monster"
    elif  isinstance(monster_data["monster_create"]["class"], BoxMonster):
        return "Throw Blocker"
    else:
        return "Damage Monster"

def get_monster_combination(monsters):
    types = [identify_monster_type(monster) for monster in monsters]
    # Sort to ensure consistent combinations
    return " + ".join(sorted(types))

def extract_monster_info(monsters):    
    # Classify the first and second monsters and extract their HP
    monsters = [(identify_monster_type(monster), monster['kwargs']['hp']) for monster in monsters]
    monsters = sorted(monsters, key=lambda x: x[0])
    
    series = [len(monsters)]
    for monster in monsters:
        series.extend(monster)
        
    return pd.Series(series)

# Apply the function to extract the required columns

# Show the resu

# Initial state setup
if 'level' not in st.session_state:
    st.session_state['level'] = pd.DataFrame(get_real_levels(full_info=True))
    st.session_state['level']["monster_combination"] = st.session_state['level']["monsters"].apply(get_monster_combination)
    st.session_state['level'][['num_mons', 'monster_1', 'monster_1_hp', 'monster_2', 'monster_2_hp']] = st.session_state['level']['monsters'].apply(extract_monster_info)

    st.session_state['level'].to_csv('level.csv', index=False)

if 'old' not in st.session_state:
    st.session_state['old'] = {
        "num_level": 0,
        "avg_win_rate": 0,
        "avg_hit_rate": 0,
        "avg_damage_per_hit": 0,
        "remain_hp_monster": 0,
    }

store_dir = "_saved_test"

# Function to load all CSVs from a given folder
def load_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df.sort_values(by=["realm_id", "node_id"])
        return df

# Plotting functions
def draw_line(x_axis, y_axis, color, label, x_tick_label_format):
    fig, ax  = plt.subplots(figsize=(5, 5))
    ax.plot(x_axis, y_axis, label=label, color=color)
    ax.set_xticks(x_axis, [f"{x_tick_label_format} {realm}" for realm in x_axis], rotation=60, ha='right')
    ax.legend(loc='best', ncols=1)
    ax.set_ylim([0, 100])
    ax.grid(True, linestyle='--', alpha=0.7)
    for x, y in zip(x_axis, y_axis):
        ax.annotate(f'{y}%', xy=(x, y), color='black', ha='center', va='bottom', xytext=(0, 1), textcoords='offset points')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    return fig

def draw_bar(df, realm, x_axis, type):
    fig, ax = plt.subplots(4, 1, figsize=(20, 9), sharex=True)
    sub_df = df[df[type] == realm]
    sub_df.loc[:, "win_rate"] = sub_df["win_rate"] * 100
    sub_df.loc[:, "hit_rate"] = sub_df["hit_rate"] * 100
    sub_df.loc[:, "remain_hp_monster"] = sub_df["remain_hp_monster"] * 100
    axes = sub_df.plot.bar(x=x_axis, y=['win_rate', 'hit_rate', 'remain_hp_monster'], subplots=True, ax=ax[0: 3], rot=65, title=f'{type.replace("_", " ")}: {realm}')
    for axs in axes:
        axs.yaxis.set_major_formatter(ticker.PercentFormatter())
        axs.set_ylim([0, 100])
        axs.grid(True, linestyle='--', alpha=0.7)
        for bar in axs.patches:
            if bar.get_height() < 100 and bar.get_height() > 0:
                axs.annotate(f'{format(bar.get_height(), ".0f")}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    draw_bar_monster(sub_df, realm, ax[3], x_axis)
    return fig

def draw_bar_monster(df, realm, ax, x_axis):

    colors = ['g' if m >= 0 else 'r' for m in df['max_step_&_monster_hp'].tolist()]
    ax = df.plot.bar(x=x_axis, y='max_step_&_monster_hp', ax=ax, rot=65, title=f"Difference between Max Step and Monster HP", color=colors)
    ax.legend(handles=[Patch(facecolor='red'), Patch(facecolor='green')], labels=["Below 0 (Hard)", "Above 0 (Easier)"], loc='best', fontsize=10)  # Clear legend
    ax.grid(True, linestyle='--', alpha=0.7)
    for bar in ax.patches:
        annotation_text = str(format(bar.get_height(), '.0f'))
        xytext_offset = (0, -8) if bar.get_height() < 0 else (0, 8)
        ax.annotate(annotation_text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=xytext_offset, textcoords='offset points')
    ax.annotate("Higher is easier to win the level.", xy=(0.5, -0.4), xycoords='axes fraction', ha='center', va="center", fontsize=13)

# Function for additional metrics
def draw_additional_plots(df, realm, x_axis, type):

    sub_df = df[df[type] == realm]
    fig, ax = plt.subplots(5, 1, figsize=(20, 9), sharex=True)
    
    metrics = ['pu_on_box_rate', 'num_pu_move_rate', 'num_match_move_rate', 'num_pu_hit_rate', 'num_match_hit_rate']
    titles = ['Power-Up on Box Rate', 'Number of Power-Up Moves Rate', 'Number of Match Moves Rate', 'Power-Up Hit Rate', 'Match Hit Rate']
    for metric in metrics:
        sub_df.loc[:, metric] = sub_df[metric] * 100
        if metric == 'pu_on_box_rate':
            sub_df.loc[:, metric] = sub_df[metric] / sub_df["hit_rate"]
    axes = sub_df.plot.bar(x=x_axis, y=metrics, subplots=True, ax=ax, rot=65, title=f'{type.replace("_", " ")}: {realm}')
    for axs, title in zip(axes, titles):
        axs.yaxis.set_major_formatter(ticker.PercentFormatter())
        axs.set_ylim([0, 100])
        axs.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in axs.patches:
            if bar.get_height() < 100 and bar.get_height() > 0:
                axs.annotate(f'{format(bar.get_height(), ".0f")}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points')

        axs.set_title(f"Realm {realm} - {title}")
        axs.set_ylabel(f"{title} (%)")
        

    return fig
            
def draw_plot(df):
    realms = df['realm_id'].unique().tolist()
    realms = sorted(realms)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['win_rate'].mean() * 100).round(1).tolist(), color="blue", label="Win rate", x_tick_label_format="realm")
        with col1:
            st.pyplot(fig)
        
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['hit_rate'].mean() * 100).round(1).tolist(), color='red', label="Hit rate", x_tick_label_format="realm")
        with col2:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['remain_hp_monster'].mean() * 100).round(1).tolist(), color='teal', label="Remaining HP of Monster (Lose Only)", x_tick_label_format="realm")
        with col3:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=np.arange(len(realms)), y_axis=(df.groupby(by=['realm_id'])['avg_damage_per_hit'].mean() * 100).round(1).tolist(), color='teal', label="AVG Damage per Hit", x_tick_label_format="realm")
        with col4:
            st.pyplot(fig)
    st.header("Realm Win Rate")
    
    level_df = st.session_state['level']
    level_df['max_step_&_monster_hp'] = level_df.apply(lambda row: row['max_step'] - sum([mon['kwargs']['hp'] for mon in row['monsters']]), axis=1)
    df = pd.merge(df, level_df, on=['realm_id', 'node_id'], how="inner")
    # this is the line merge df to have the above csv
    for realm in realms:
        with st.columns([1,8, 1])[1]:
            fig = draw_bar(df, realm, "node_id", "realm_id")
            st.pyplot(fig, use_container_width=False)
            try:
                fig = draw_additional_plots(df, realm, "node_id", "realm_id")
                st.pyplot(fig, use_container_width=False)
            except:
                continue
    return df

def draw_monster_type_plot(df):
    st.header("Monster Common Statistics")
    
    df["monster_combination"] = df["monsters"].apply(get_monster_combination)
    df["realm__node"] = df.apply(lambda row: f'{row["realm_id"]}__{row["node_id"]}', axis = 1)
    
    monster_combination = df['monster_combination'].unique().tolist()
    monster_combination = sorted(monster_combination)
    print(monster_combination)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        fig = draw_line(x_axis=monster_combination, y_axis=(df.groupby(by=['monster_combination'])['win_rate'].mean() * 100).round(1).tolist(), color="blue", label="Win rate", x_tick_label_format="")
        with col1:
            st.pyplot(fig)
        
        fig = draw_line(x_axis=monster_combination, y_axis=(df.groupby(by=['monster_combination'])['hit_rate'].mean() * 100).round(1).tolist(), color='red', label="Hit rate", x_tick_label_format="")
        with col2:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=monster_combination, y_axis=(df.groupby(by=['monster_combination'])['remain_hp_monster'].mean() * 100).round(1).tolist(), color='teal', label="Remaining HP of Monster (Lose Only)", x_tick_label_format="")
        with col3:
            st.pyplot(fig)
            
        fig = draw_line(x_axis=monster_combination, y_axis=(df.groupby(by=['monster_combination'])['avg_damage_per_hit'].mean() * 100).round(1).tolist(), color='teal', label="AVG Damage per Hit", x_tick_label_format="")
        with col4:
            st.pyplot(fig)
    st.header("Monster Win Rate")
    
    for monster_comb in monster_combination:
        with st.columns([1,8, 1])[1]:
            fig = draw_bar(df, monster_comb, "realm__node", "monster_combination")
            st.pyplot(fig, use_container_width=False)
            try:
                fig = draw_additional_plots(df, monster_comb, "realm__node", "monster_combination")
                st.pyplot(fig, use_container_width=False)
            except:
                continue


# Common Statistics Section
def draw_common_stat(df):
    st.subheader(f"Test on {df['num_games'].max()} games per level ({len(df)} levels)")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_win_rate = df['win_rate'].mean() * 100
    avg_hit_rate = df['hit_rate'].mean() * 100
    avg_damage_per_hit = df['avg_damage_per_hit'].mean() * 100
    remain_hp_monster = df[df['remain_hp_monster'] > 0]["remain_hp_monster"].mean() * 100
    
    col1.metric("Average Win Rate", f"{avg_win_rate:.2f}%", delta=f"{(avg_win_rate - st.session_state['old']['avg_win_rate']):.2f}%")
    col2.metric("Average Hit Rate", f"{avg_hit_rate:.2f}%", delta=f"{(avg_hit_rate - st.session_state['old']['avg_hit_rate']):.2f}%")
    col3.metric("Remaining HP of Monster (Lose Only)", f"{remain_hp_monster:.2f}%", delta=f"{(remain_hp_monster - st.session_state['old']['remain_hp_monster']):.2f}%", delta_color="inverse")
    col4.metric("Average Damage per Hit", f"{avg_damage_per_hit:.2f}%", delta=f"{(avg_damage_per_hit - st.session_state['old']['avg_damage_per_hit']):.2f}")

    # Update old state
    st.session_state['old'].update({
        "avg_win_rate": avg_win_rate,
        "avg_hit_rate": avg_hit_rate,
        "avg_damage_per_hit": avg_damage_per_hit,
        "remain_hp_monster": remain_hp_monster,
    })



# Main Streamlit app
st.title("M3 Testing Metrics")

# Input folder name
folder_path = st.selectbox("Select the model:", options=[d for d in sorted(os.listdir(store_dir)) if os.path.isdir(os.path.join(store_dir, d))])

if folder_path:
    # Load CSV and display
    df = load_csv_files(os.path.join(store_dir, folder_path))
    st.success(f"Loaded {len(df)} rows.")

    # Common stats
    st.header("Common Statistics", divider=True)
    draw_common_stat(df)

    # Plot Realm-level charts
    st.header("Realm-Level Metrics", divider=True)
    merge_df = draw_plot(df)

    draw_monster_type_plot(merge_df)
    # st.pyplot(fig, use_container_width=True)

    # Display the full data table
    st.subheader("Full Data")
    st.dataframe(df, use_container_width=True)




