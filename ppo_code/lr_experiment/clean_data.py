'''
This processes the raw adult data (adult.data and adult.test from https://archive.ics.uci.edu/ml/datasets/adult) 
which reside in data/adult_dataset/raw and puts cleaned data table in data/adult_data/adult_clean_full.csv

While the raw dataset has 48,842 records (rows) with 15 columns (14 features and the target column), 
we drop rows that have missing values leaving 45,222 rows. 
The target column (to be predicted from the other 14 features) is binary, 
indicating whether the adjusted gross income is above/less or equal to $50K. 

Column `fnlwgt` is numeric and tells the proportion of the population that has the same set of features. 
In short, this is not a feature of an individual, (and so we should not use it as a feature for the prediction task). 
On the other hand, since it tells how often this row of data appears, it does influence our optimization function. 
It is left in the dataset when this script is run. 

Five columns (`age, capital_loss, captial_gain, education_num, hours_per_week`) are numeric, and are left as is. 
Two columns (`sex`, and the target column) are binary and the remaining eight columns are categorical with more than two values.  
We have converted all binary categorical columns to a numerical binary values {0,1}, 
and categorical columns with more than two values into one-hot encoding 
(this exands a single column with k values to k columns each with values in {0,1}.).  

After this cleaning, the data has 45222 rows, and 105 columns (103 features, 1 target, 1 fnlwgt column). 
Not accounting for fnlwgt counts, the simple class bias is approximately 24.8%/75.2%, 
so the constant "0" classifier  achieves 0% recall and 75.2% accuracy, 
while the constant "1" classifier acheives 100% recall and 24.8% accuracy. 
'''

#%%
import pandas as pd 
from pathlib import Path 
from matplotlib import pyplot as plt 
# from sklearn.preprocessing import LabelEncoder
import numpy as np 
import sys 

# add our ./code folder to sys.path so we can import our modules: 
# sys.path.append(Path(".", "code").absolute().as_posix())
# from plotting_utils import * # our own plotting utilities
# from data_utils import * # our own data utilities

# add our ./code/lr_experiment/ folder so we can find our config file 
sys.path.append(Path(".", "code", "lr_experiment").absolute().as_posix())
from config import *


# %% read in raw data (two files), dropping rows w/ missing values, and then make a single table
df_train = pd.read_csv(ADULT_DATA_PATH, header = None, names = COLS,  skipinitialspace = True, na_values= "?").dropna()
df_test = pd.read_csv(ADULT_TEST_PATH, header = None, names = COLS, skipinitialspace=True, skiprows=1, na_values= "?").dropna()
df_full = pd.concat([df_train, df_test], axis = 0).reset_index()
# df_full = df_full[df_full.columns.difference(['fnlwgt'])]

# %%  fix target typos: 
df_full = df_full.replace('<=50K.', '<=50K')
df_full = df_full.replace('>50K.', '>50K')

# %% get lists of col names that are categorical / numerical
num_x = df_full.select_dtypes(include=['int64', 'float64']).columns
cat_x = df_full.select_dtypes(include=['object', 'bool']).columns
bin_x = [ c for c in cat_x if len(df_full[c].unique())==2]
cat_x = [c for c in cat_x if c not in bin_x] # remove binary from cat_x

# %%  make new dataframe with numerical values
df = df_full[num_x.difference(['index'])]


# %%  add one_hot encoding for categorical, non-binary variables 
for c in cat_x: 
    x = pd.get_dummies(df_full[c], prefix = c)
    df = pd.concat([df, x], axis = 1)


#%% add binary encoding for binary columns 
for c in bin_x: 
    x = pd.get_dummies(df_full[c])
    last_col = x.columns[-1]
    if c == "target": 
        df.loc[:, 'target'] = x[last_col]
    else: 
        df.loc[:, c + "_" + last_col] = x[last_col]


# %% write to csv 
df.to_csv(ADULT_CLEAN_FULL_PATH)