import pandas as pd
import numpy as np

df = pd.read_csv('./data/sp500.csv',index_col='Date',parse_dates=True).sort_index()
df['return'] = np.log(df['Close']).diff()

print(df.head())

import matplotlib.pyplot as plt
df['return'].plot()
plt.show()