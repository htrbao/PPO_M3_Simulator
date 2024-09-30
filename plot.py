#!/usr/bin/env python
# coding: utf-8

# In[42]:


import matplotlib.pyplot as plt
import pandas as pd


# In[43]:


file = "./_saved_csv/new_cal_10layers_32channels_2e-05_4096_not_share_20240926/csv_log_count_win.csv"


# In[44]:


max_cols = 74
data = pd.read_csv(file, header=None).iloc[:,:max_cols]

# data = data.div(data.sum(axis=1), axis=0)


# In[45]:


data.sum(axis=1)


# In[46]:


data.mean(axis=1)


# In[47]:


ax = data[:].plot.bar(figsize=(20, 8), grid=True)
ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
ax.set_ylabel('Num Game Win')
ax.set_xlabel('Trained Step')


# In[48]:


import os
ax.figure.savefig(os.path.join(os.path.dirname(file), "img.jpg"))

