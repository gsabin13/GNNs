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

df_reddit = df.loc[df['Dataset']=='reddit']
df_arxiv = df.loc[df['Dataset']=='ogbn-arxiv']
df_products = df.loc[df['Dataset']=='ogbn-products']
df_meta = df.loc[df['Dataset']=='meta']
df_arctic25 = df.loc[df['Dataset']=='arctic25']
df_oral = df.loc[df['Dataset']=='oral']

# gpu=8

#df_reddit = df_reddit.loc[df_reddit['GPU']==2]
#df_arxiv = df_arxiv.loc[df_arxiv['GPU']==2]
#df_products = df_products.loc[df_products['GPU']==2]
#df_meta = df_meta.loc[df_meta['GPU']==2]
#df_arctic25 = df_arctic25.loc[df_arctic25['GPU']==2]
#df_oral = df_oral.loc[df_oral['GPU']==2]


sns.set_style("whitegrid")
for xaxis in ['Time']:
    for i, df in enumerate([df_arxiv, df_reddit, df_products, df_meta, df_arctic25, df_oral]):
        #df_ = df.loc[(df['Method-GPU']=='GraphSAINT-DGL')| (df['Method-GPU']=='GraphSAINT-DGL')|(df['Method-GPU']=='GCN-RDM') ]
        #df_ = df.loc[(df['Method-GPU']=='GraphSAINT-DGL')| (df['Method-GPU']=='GraphSAINT-DGL')|(df['Method-GPU']=='GCN-RDM') ]
        #g = sns.lineplot(data=df, x=xaxis, y='Acc', hue='Method-GPU', palette='muted',hue_order=["GraphSAINT-DGL","GraphSAINT-RDM","GCN-RDM"])
        g = sns.lineplot(data=df, x=xaxis, y='Acc', hue='Method-GPU', palette='muted')
        plt.xlabel(xaxis, fontsize=20);
    #else:
    #    g = sns.lineplot(data=df, x='Epoch', y='Test', hue='Method', palette='muted')
    #    plt.xlabel('Epoch', fontsize=20);
        pref = 'jul10'
        #pref = 'graphsaint_comparison_figs'
        #g.set(xlim=(0, 500))
        if i == 0:
            g.set(ylim=(0.2, 0.75))
            g.set(xlim=(0, 50))
            output = f'{pref}/arxiv_saint_{xaxis}.pdf'
        elif i==1: 
            g.set(ylim=(0.8, 0.97))
            output = f'{pref}/reddit_saint_{xaxis}.pdf'
            g.set(xlim=(0, 40))
        elif i==2:
            g.set(ylim=(0.2, 0.9))
            output = f'{pref}/products_saint_{xaxis}.pdf'
            g.set(xlim=(0, 40))
        elif i==3:
            output = f'{pref}/meta_saint_{xaxis}.pdf'
            g.set(xlim=(0, 100))
        elif i==4:
            output = f'{pref}/arctic25_saint_{xaxis}500.png'
            g.set(xlim=(0, 100))
        elif i==5:
            output = f'{pref}/oral_saint_{xaxis}500.png'
            g.set(xlim=(0, 100))
    #    if i == 0:
    #        g.set(ylim=(0, 0.8))
    #        g.set(xlim=(0, 50))
    #        output = 'arxiv.png'
    #    elif i==1: 
    #        g.set(ylim=(0.8, 0.97))
    #        output = 'reddit.png'
    #    else:
    #        g.set(ylim=(0, 0.85))
    #        output = 'products.png'
        plt.ylabel('Test Accuracy', fontsize=20);
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(output)
        plt.clf()