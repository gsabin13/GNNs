import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import seaborn as sns

filename = sys.argv[1]
df = pd.read_csv(filename)
df.dropna(axis=1, inplace=True)
col = list(df.columns)
print(col)
sns.set(rc = {'figure.figsize':(60,60)})
#g = sns.barplot(data=df, x='#GPU', y='Epoch time', hue='Method', palette='Paired')
g = sns.lineplot(data=df, x='Time', y='Acc', hue='Method-GPU', palette='muted',hue_order=["GraphSAINT-DGL","GraphSAINT-RDM","GCN-RDM"],linewidth=5)
leg = plt.legend(fontsize=70, ncol=1,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0)
for line in leg.get_lines():
    line.set_linewidth(15.0)
#plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
plt.savefig('legend_fig11.pdf')
plt.cla()
