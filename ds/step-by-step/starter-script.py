import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

seed = 42
np.random.seed(seed)

"""

```
palette="Set3"
palette="husl"
```

"""


