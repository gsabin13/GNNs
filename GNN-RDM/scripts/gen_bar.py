import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import seaborn as sns

filename = sys.argv[1]
output = sys.argv[2]

df = pd.read_csv(filename)
df.dropna(axis=1, inplace=True)
col = list(df.columns)
print(col)
sns.set_style("whitegrid")
sns.barplot(data=df, x='Number of GPUs', y='Training Throughput (epochs/s)', hue='Method', palette='Paired')
plt.xlabel('Number of GPUs', fontsize=20);
plt.ylabel('Training Throughput (epochs/s)', fontsize=20);
plt.legend(fontsize=14, ncol=2,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
if 'arxiv' in filename:
        t = 'OGB-Arxiv'
elif 'pro' in filename:
        t = 'OGB-Products'
elif 'redd' in filename:
        t = 'Reddit'
elif 'goo' in filename:
        t = 'Web-Google'
elif 'ork' in filename:
        t = 'Com-Orkut'
elif 'mag' in filename:
        t = 'OGB-MAG'
plt.savefig(output)

