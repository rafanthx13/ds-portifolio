STARTER SCRIPT

=======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format

sns.set(style="whitegrid")

plt.style.use('seaborn')
sns.set_palette("Set3") # default seabon: deep || default matplotlib: tab10 || Set1, Set2, Set3, Paired, muted, Accent, Spectral, CMRmap # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

plt.style.use('seaborn')

seed = 42
np.random.seed(seed)

=======================================================================

file_path = '/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv'
df = pd.read_csv(file_path)
print("DataSet = {} rows and {} columns".format(df.shape[0], df.shape[1]))
print("Columns:", df.columns.tolist())
df.head()

=======================================================================

quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']
print("Qualitative Variables: (Numerics)", "\n\n=>", qualitative,
      "\n\nQuantitative Variable: (Strings)\n=>", quantitative)

=======================================================================
