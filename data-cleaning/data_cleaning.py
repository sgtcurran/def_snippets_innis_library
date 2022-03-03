#%%
import pandas as pd

df = pd.DataFrame
#%%
#data cleaning 
#remove currency 
def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)

# Methods to replace NaN values with zeros in Pandas DataFrame:

# For the whole DataFrame using pandas:
df.fillna(0) 
# The dataframe.replace() function in Pandas can be defined as a simple method 
# used to replace a string, regex, list, dictionary etc. in a DataFrame.
df.replace(np.nan, 0) 

#For one column using pandas:
df['DataFrame Column'] = df['DataFrame Column'].fillna(0)
# For one column using numpy:
df['DataFrame Column'] = df['DataFrame Column'].replace(np.nan, 0)

# replace str in column(s) using pandas without creating a dictionary file:
df.columns[df.columns == 'str'] = num

# drops duplicates in pd.DataFrame and np.arrays 
df.drop_duplicates()

# keeping wanted columns by list and else removing unwanted columns not listed from DataFrame 
df = df.drop(df.columns.difference(['columns_name'], sort = False), axis=1)


# %%
