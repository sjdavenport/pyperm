# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:03:30 2023

@author: 12SDa
"""

for i in range(10000):
    loader(i, 10000)

# %%
for i in range(total):
    time.sleep(0.01)  # Simulate a long computation
    progress_bar(i + 1, total, prefix='Progress:',
                 suffix='Complete', length=50)
