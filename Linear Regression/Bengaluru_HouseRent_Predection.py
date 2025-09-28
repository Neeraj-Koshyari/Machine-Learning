import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Data/Bengaluru_House_Data.csv')
df.head()

# Drop columns that are not needed for modelling
df1 = df.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)
df1.head()

# Inspect and remove missing values
df1.isnull().sum()
df2 = df1.dropna()
df2.isnull().sum()

# Extract numeric BHK from the 'size' text field
df2['BHK'] = df2['size'].apply(lambda x: int(x.split()[0]))
df2.head()

# Convert total_sqft entries like "2100 - 2850" to a numeric value (average)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# NOTE: the code below assumes df3 exists as the dataframe you want to transform.
# If following from the previous variables, you may want to set df3 = df2.copy() before this apply.
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)

# Drop rows with NaNs introduced by conversion (if any)
df3.dropna()

# Create price per sqft column
df3['price_per_sqft'] = df3['price'] * 100000 / df3['total_sqft']

# Normalize location strings and aggregate location statistics
df4['location'] = df4['location'].apply(lambda x: x.strip())
group = df4.groupby('location')
location_stats = group['location'].count().sort_values(ascending=False)

# Group locations with very few data points into 'other'
location_less_then_10 = location_stats[location_stats <= 10]
df4['location'] = df4['location'].apply(lambda x: 'other' if x in location_less_then_10 else x)
len(df4['location'].unique())

# Remove entries with unrealistic small area per BHK (e.g., < 300 sqft/BHK)
df4[df4['total_sqft'] / df4['BHK'] < 300]
df5 = df4[~(df4['total_sqft'] / df4['BHK'] < 300)]
df5.head()

# Remove price-per-sqft outliers per location (±1 std)
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subDf in df.groupby('location'):
        m = np.mean(subDf['price_per_sqft'])
        st = np.std(subDf['price_per_sqft'])
        need_row = subDf[(subDf['price_per_sqft'] > (m - st)) & (subDf['price_per_sqft'] < (m + st))]
        df_out = pd.concat([df_out, need_row], axis=0, ignore_index=True)
    return df_out

df6 = remove_pps_outliers(df5)

# Remove BHK outliers where a higher-BHK apartment is priced lower than mean of lower-BHK in same location
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df7 = remove_bhk_outliers(df6)

# Remove rows with unrealistic number of bathrooms relative to BHK
df8 = df7[df7['bath'] < (df7['BHK'] + 2)]
df8.head()

# Drop columns not needed for final modelling and create one-hot encodings for location
df9 = df8.drop(['size', 'price_per_sqft'], axis=1)
df9.head()

dummies = pd.get_dummies(df9['location']).map(lambda x: 1 if x is True else 0).drop('other', axis=1)
dummies.head()

df10 = pd.concat([df9, dummies], axis=1)
df10.head()

df11 = df10.drop('location', axis=1)
df11.head()

# Prepare feature matrix X and target y
x = df11.drop('price', axis=1)
y = df11['price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# NOTE: the variable `result` is not defined in your snippet.
# You probably want to compute predictions first, e.g.:
# result = lr.predict(x_test)
# Then compute r2_score:
from sklearn.metrics import r2_score
# r2_score(y_test, result)

# Helper: predict price for a single input (expects lr and x to be defined)
def predict_price(total_sqft, bath, BHK, location):
    loc_idx = np.where(x.columns == location)[0][0]
    arr = np.zeros(len(x.columns))
    arr[0] = total_sqft
    arr[1] = bath
    arr[2] = BHK
    if loc_idx >= 0:
        arr[loc_idx] = 1
    return lr.predict([arr])[0]

print(predict_price(1630.0, 3.0, 3, '1st Block Jayanagar'))
