import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import seaborn as sns

filename = sys.argv[1]
#output = sys.argv[2]
#t = sys.argv[3]=='t'

df = pd.read_csv(filename)
df.dropna(axis=1, inplace=True)
col = list(df.columns)
print(col)

df_reddit = df.loc[df['Dataset']=='Reddit']
df_arxiv = df.loc[df['Dataset']=='ogbn-arxiv']
df_products = df.loc[df['Dataset']=='ogbn-products']
df_meta = df.loc[df['Dataset']=='meta']
df_arctic25 = df.loc[df['Dataset']=='arctic25']
df_oral = df.loc[df['Dataset']=='oral']

sns.set_style("whitegrid")
for i, df in enumerate([df_arxiv, df_reddit, df_products, df_meta, df_arctic25, df_oral]):
    g = sns.barplot(data=df, x='#GPU', y='Epoch time', hue='Method', palette='Paired')
    #g = sns.barplot(data=df, x='#GPU', y='Training Throughput (epochs/s)', hue='Method', palette='Paired')
    plt.xlabel('Number of GPUs', fontsize=20);
    plt.ylabel('Training Throughput (epochs/s)', fontsize=20);
    plt.legend(fontsize=14, ncol=2,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0)
    plt.tick_params(axis='both', which='major', labelsize=14)
    #plt.tight_layout()
    #g = sns.lineplot(data=df, x='Time', y='Acc', hue='Method', palette='muted')
    #plt.xlabel('Time (seconds)', fontsize=20);
#else:
#    g = sns.lineplot(data=df, x='Epoch', y='Test', hue='Method', palette='muted')
#    plt.xlabel('Epoch', fontsize=20);
    #if 'arxiv' in filename:
    #        t = 'OGB-Arxiv'
    #elif 'pro' in filename:
    #        t = 'OGB-Products'
    #elif 'redd' in filename:
    #        t = 'Reddit'
    #elif 'goo' in filename:
    #        t = 'Web-Google'
    #elif 'ork' in filename:
    #        t = 'Com-Orkut'
    #elif 'mag' in filename:
    #        t = 'OGB-MAG'
    midlayer = sys.argv[2]

    if i == 0:
        #g.set(ylim=(0, 0.8))
        #g.set(xlim=(0, 50))
        output = f'arxiv_tpt_{midlayer}.png'
    elif i==1: 
        #g.set(ylim=(0.8, 0.97))
        output = f'reddit_tpt_{midlayer}.png'
    elif i==2:
        #g.set(ylim=(0, 0.85))
        output = f'products_tpt_{midlayer}.png'
    elif i==3:
        output = f'meta_tpt_{midlayer}.png'
    elif i==4:
        output = f'arctic25_tpt_{midlayer}.png'
    elif i==5:
        output = f'oral_tpt_np_{midlayer}.png'
    #plt.ylabel('Test Accuracy', fontsize=20);
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(output)
    plt.clf()