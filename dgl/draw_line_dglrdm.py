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
#df_products = df.loc[df['Dataset']=='ogbn-products']
df_meta = df.loc[df['Dataset']=='meta']
df_arctic25 = df.loc[df['Dataset']=='arctic25']
df_oral = df.loc[df['Dataset']=='oral']
#df_mag = df.loc[df['Dataset']=='ogbn-mag']

# gpu=8

#df_reddit = df_reddit.loc[df_reddit['GPU']==2]
#df_arxiv = df_arxiv.loc[df_arxiv['GPU']==2]
#df_products = df_products.loc[df_products['GPU']==2]
#df_meta = df_meta.loc[df_meta['GPU']==2]
#df_arctic25 = df_arctic25.loc[df_arctic25['GPU']==2]
#df_oral = df_oral.loc[df_oral['GPU']==2]

sns.set_style("whitegrid")
from matplotlib import rcParams
print(rcParams['figure.figsize'])
rcParams['figure.figsize'] = 10,6
for xaxis in ['Epoch','Time']:
    #for i, df in enumerate([df_arxiv, df_reddit, df_products, df_meta, df_arctic25, df_oral]):
    #for i, df in enumerate([df_arxiv, df_reddit, df_products]):
    for i, df in enumerate([df_meta, df_arctic25, df_oral, df_reddit,df_arxiv]):
        #df_ = df.loc[(df['Method-GPU']=='GraphSAINT-DGL')| (df['Method-GPU']=='GraphSAINT-DGL')|(df['Method-GPU']=='GCN-RDM') ]
        #df_ = df.loc[(df['Method-GPU']=='GraphSAINT-DGL')| (df['Method-GPU']=='GraphSAINT-DGL')|(df['Method-GPU']=='GCN-RDM') ]
        #g = sns.lineplot(data=df, x=xaxis, y='Acc', hue='Method-GPU', palette='muted',hue_order=["GraphSAINT-DGL","GraphSAINT-RDM","GCN-RDM"],linewidth=3)
        g = sns.lineplot(data=df, x=xaxis, y='Acc', hue='Method', palette='Paired', linewidth=2)
        plt.xlabel(xaxis, fontsize=28);
    #else:
    #    g = sns.lineplot(data=df, x='Epoch', y='Test', hue='Method', palette='muted')
    #    plt.xlabel('Epoch', fontsize=20);
        pref = 'aug10'
        #pref = 'graphsaint_comparison_figs'
        #g.set(xlim=(0, 500))
        if i == 0:
            #g.set(ylim=(0.2, 0.75))
            #g.set(xlim=(0, 50))
            output = f'{pref}/airways_{xaxis}.png'
            #plt.legend(title='airways',fontsize=24)
            plt.legend(title='airways',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #plt.setp(g.get_legend().get_title(), fontsize='20')
        elif i==1: 
#            g.set(ylim=(0.87, 0.97))
            output = f'{pref}/arctic25_{xaxis}.png'
            #plt.legend(title='arctic25',fontsize=24)
            #plt.setp(g.get_legend().get_title(), fontsize='20')
            plt.legend(title='arctic25',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #g.set(xlim=(0, 250))
        elif i==2:
            #g.set(ylim=(0.2, 0.9))
            output = f'{pref}/oral_{xaxis}.png'
            #plt.legend(title='oral',fontsize=24)
            plt.legend(title='oral',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #g.set(xlim=(0, 40))
        elif i==3:
            output = f'{pref}/reddit_{xaxis}.png'
            #plt.legend(title='reddit',fontsize=24)
            plt.legend(title='reddit',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #g.set(xlim=(0, 100))
        elif i==4:
            output = f'{pref}/arxiv_{xaxis}.png'
            plt.legend(title='arxiv',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #g.set(xlim=(0, 100))
        elif i==5:
            g.set(ylim=(0.4, 0.9))
            output = f'{pref}/oral_{xaxis}.png'
            #plt.legend(title='arxiv',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            #g.set(xlim=(0, 100))
#        elif i==6:
#            output = f'{pref}/mag_{xaxis}.pdf'
#            g.set(xlim=(0, 150))
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
        plt.ylabel('Test Accuracy', fontsize=28);
#        plt.legend([],[], frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.tight_layout()
        plt.savefig(output)
        plt.clf()