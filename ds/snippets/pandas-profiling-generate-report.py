# Import libs
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

# Import dataFrame
df = pd.read_csv("2004-2019.tsv", delimiter="\t")

# Generate pandas profiling
pfr = ProfileReport(df)
pfr.to_file("/home/rhavel/Documentos/example.html")

