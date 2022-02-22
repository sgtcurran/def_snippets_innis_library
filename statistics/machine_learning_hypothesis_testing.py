#%%
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7)
plt.rcParams['image.cmap'] = 'viridis'

%matplotlib inline

tfd = tfp.distributions
tfb = tfp.bijectors

# %%

# %%
# Contingency table creator & Chi-Square Test Calculator - used to have two categorical variable from a 
# single population. to determine whether there is a significant association between the two variables.

# readme : all you have to do is put in the df['column'], df['column'] and the code will do the rest. 

contingency_table=pd.crosstab(df['column'],df['column'])
print('contingency_table :-\n',contingency_table)

#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
# alpha = confadance level .95
alpha = 0.05

from scipy.stats import chi2

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)
#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

#%%
from scipy.stats import ttest_1samp
import numpy as np

# t-test - one sample 
data = data #sample 1

data_mean = np.mean(data)
print(data_mean)

tset, pval = ttest_1samp(data, data_mean)
# alpha value is 0.05 or 5%
print("p-values", pval) 
if pval < 0.05:   
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")
#%%
from scipy.stats import ttest_ind

# t-test two sample of two independent samples or two independent groups

data1 = data #sample 1
data2 = data #sample 2

print(data1)
print("data2 data :-\n")
print(data2)
data1_mean = np.mean(data1)
data2_mean = np.mean(data2)

print("data1 mean value:",data1_mean)
print("data2 mean value:",data2_mean)

data1_std = np.std(data1)
data2_std = np.std(data2)

print("data1 std value:",data1_std)
print("data2 std value:",data2_std)

ttest,pval = ttest_ind(data1,data2)
print("p-value",pval)

if pval <0.05:
    print("we reject null hypothesis")
else:
    print("we accept null hypothesis")