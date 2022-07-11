import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import seaborn as sns

filename = sys.argv[1]
output = sys.argv[2]
t = sys.argv[3]=='t'


df = pd.read_csv(filename)
df.dropna(axis=1, inplace=True)
col = list(df.columns)
print(col)
sns.set_style("whitegrid")
if t:
    g = sns.lineplot(data=df, x='Time', y='Test', hue='Method', palette='muted')
    plt.xlabel('Time (seconds)', fontsize=20);
else:
    g = sns.lineplot(data=df, x='Epoch', y='Test', hue='Method', palette='muted')
    plt.xlabel('Epoch', fontsize=20);
#if 'rxiv' in sys.argv[1]:
#    g.set(xlim=(0, 100))
#else:
#    g.set(xlim=(0, 150))

if 'EDD'.lower() in sys.argv[1].lower():
    g.set(ylim=(0.88, 0.97))
elif 'fl' in sys.argv[1]:
    g.set(ylim=(0.3, 0.55))
else:
    g.set(ylim=(0.5, 0.8))
plt.ylabel('Test Accuracy', fontsize=20);
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig(output)