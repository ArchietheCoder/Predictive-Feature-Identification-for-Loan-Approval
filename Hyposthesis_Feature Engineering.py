#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# 
# #### Predictive Feature Identification for Loan Approval using Hypothesis Testing and Feature Engineering
# 
# In the context of a loan dataset, the goal is to identify key features that robustly predict whether an upcoming candidate is likely to be a good candidate for loan approval. The task involves employing hypothesis testing and feature engineering techniques to enhance the predictive capabilities of the model.
# 
# # Objective:
# To determine and create features that significantly contribute to predicting the creditworthiness of individuals applying for loans, using statistical hypothesis testing and innovative feature engineering.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
from scipy.stats import chi2_contingency, levene, kruskal, kstest
from statsmodels.graphics.gofplots import qqplot


# In[2]:


df=pd.read_csv('D:\Scaler\Scaler\Hypothesis Testing\Data-Set\loan.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df['Dependents'].value_counts()


# In[9]:


df['Loan_Status'].value_counts()


# In[10]:


#sns.countplot(df["Loan_Status"])
df['Loan_Status'] = df['Loan_Status'].astype(str)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Status Count")
plt.show()


# ## Appicant Income

# In[11]:


sns.histplot(df['ApplicantIncome'])


# In[12]:


sns.histplot(np.log(df['ApplicantIncome']), kde = True)


# In[13]:


sns.kdeplot(np.log(df['ApplicantIncome']))


# In[14]:


qqplot(np.log(df['ApplicantIncome']), line='s')
plt.show()


# ### Applicant Income is Important/ Good Predictor?

# In[15]:


df.groupby("Loan_Status").mean(numeric_only=True)


# In[ ]:





# ##### Ho : Loan Status is Accepted = Loan Status is Rejected
# ##### Ha : Loan Status is Accepted != Loan Status is Rejected

# In[16]:


df_acc = df.loc[df['Loan_Status'] == 'Y']['ApplicantIncome']
df_rej = df.loc[df['Loan_Status'] == 'N']['ApplicantIncome']


# In[17]:


df_acc.mean(), df_rej.mean()


# #### Approach 1: As This is categorical vs numerical and df_acc and df_rej are independent variable, we will perform ttest_independent

# In[18]:


alpha = 0.05
_, p_val = ttest_ind(df_acc, df_rej, alternative = 'less' )
if p_val < alpha:
    print("Loan Status is Accepted != Loan Status is Rejected. Hence, There is effect of Applicant Income on Loan status")
else:   
    print("Loan Status is Accepted = Loan Status is Rejected. Hence, There is no effect of Applicant Income on Loan status")


# In[19]:


ttest_ind(df_acc, df_rej, alternative = 'two-sided' )
if p_val < alpha:
    print("Loan Status is Accepted != Loan Status is Rejected. Hence, There is effect of Applicant Income on Loan status")
else:   
    print("Loan Status is Accepted = Loan Status is Rejected. Hence, There is no effect of Applicant Income on Loan status")


# #### Approach 2 : We can plot kdeplot to analyse both the variables

# In[20]:


sns.kdeplot(df_acc)
sns.kdeplot(df_rej)


# #### Approach 3: We can do kstest

# In[21]:


_,p_value = kstest(df_acc, df_rej)
if p_val < alpha:
    print("Loan Status is Accepted != Loan Status is Rejected. Hence, There is effect of Applicant Income on Loan status")
else:   
    print("Loan Status is Accepted = Loan Status is Rejected. Hence, There is no effect of Applicant Income on Loan status")


# #### Approach 4: We can convert Applicant income into a catogorical column and then can perform chi2 test (catgorical vs categorical)

# In[22]:


bins = [0, 2500, 4000, 6000, 8000, 10000, 81000]
labels = ['Low','Average','medium', 'h1', 'h2', 'Very high']
df['Income_bins'] = pd.cut(df['ApplicantIncome'], bins = bins, labels = labels)
df.head()


# In[23]:


var1 = pd.crosstab(df['Income_bins'], df['Loan_Status'])
var1


# In[24]:


_, p_value,_, _ = chi2_contingency(var1)
if p_val < alpha:
    print("Loan Status is Accepted != Loan Status is Rejected. Hence, There is effect of Applicant Income on Loan status")
else:   
    print("Loan Status is Accepted = Loan Status is Rejected. Hence, There is no effect of Applicant Income on Loan status")


# #### Approach 4: Visualization method

# In[25]:


# bar graph of income bins and % share of loan status


# In[26]:


income_bins = pd.crosstab(df['Income_bins'], df['Loan_Status'])
income_bins.div(income_bins.sum(axis=1), axis=0).plot(kind="bar", figsize = (6,4))
plt.xlabel("ApplicantIncome")
plt.ylabel("Percentage")
plt.show()


# ### From the above graph, we can analyse that there is similar pattern for loan status for different income bins

# 

# ## Credit history

# In[27]:


sns.countplot(x=df['Credit_History'], hue= df['Loan_Status'])


# In[28]:


var2 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
var2


# In[29]:


_, p_value, _, _ =  chi2_contingency(var2)
alpha = 0.05
if p_value < alpha:
    print("There is an effect of credit history on loan status")
else:
    print("There is no effect of credit history on loan status")


# ### Now, like credit history, we will analyse all the variables with type "object"

# In[30]:


cat_col_series = df.dtypes == "object"
cat_columns = list(cat_col_series[cat_col_series].index)
cat_columns


# In[31]:


cat_columns.remove('Loan_ID')
cat_columns.remove('Dependents')
cat_columns.remove('Loan_Status')


# In[32]:


cat_columns


# In[33]:


def check_chi2Contingency(x,y,alpha):
    var = pd.crosstab(x,y)
    _, p_value, _, _ =  chi2_contingency(var)
    alpha = 0.05
    if p_value < alpha:
        print(f"Feature {col} is a good Predictor for target variable Loan Status")
    else:
        print(f"Feature {col} is a bad Predictor for target variable Loan Status")


# In[34]:


for col in cat_columns:
    check_chi2Contingency(df[col], df['Loan_Status'], 0.05)
    


# ### A well-defined set of features identified through hypothesis testing that significantly contribute to predicting loan approval.

# In[37]:


df.head(2)


# In[38]:


df['Total_Household_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']


# In[39]:


df.head(2)


# ## Total_Household_Income is a good/bad predictor

# In[55]:


df.groupby("Loan_Status").mean(numeric_only = True)


# #### H0: Loan Status is accepted = Loan Status is rejected
# #### Ha: Loan Status is accepted != Loan Status is rejected

# In[48]:


df_acc1 = df.loc[df['Loan_Status'] == 'Y'] ['Total_Household_Income']
df_rej1 = df.loc[df['Loan_Status'] == 'N'] ['Total_Household_Income']


# In[49]:


df_acc1.mean(), df_rej1.mean()


# #### Approach 1: Hypothesis testing through ttest_ind

# In[51]:


ttest_ind(df_acc1, df_rej1, alternative = 'less')


# #### Approach 2: Hypothesis testing through kdeplot

# In[52]:


sns.kdeplot(x = df['Total_Household_Income'], hue = df['Loan_Status'])


# #### Approach 3: Hypothesis testing through kstest

# In[53]:


kstest(df_acc1, df_rej1, alternative = 'less')


# In[41]:


df['EMI'] = (df['LoanAmount'] * 1000)/df['Loan_Amount_Term']


# In[42]:


df.head(2)


# In[56]:


df['Ability_to_pay_EMI'] = ((df['Total_Household_Income'] * 0.15) > df['EMI']).astype(int)


# In[57]:


df.head(2)


# In[60]:


va2 = pd.crosstab(df["Ability_to_pay_EMI"], df['Loan_Status'])
va2


# In[63]:


_, p_value, _, _ = chi2_contingency(va2)
p_value


# In[64]:


alpha = 0.05
if p_value < alpha:
    print("Loan status accepted != Loan Status rejected. Hence, Ability_to_pay_EMI is a good predictor")
else:
    print("Loan status accepted = Loan Status rejected. Hence, Ability_to_pay_EMI is a bad predictor")


# 
# ### New engineered feature - "Ability_to_pay_EMI" designed to enhance the model's predictive power.

# In[ ]:




