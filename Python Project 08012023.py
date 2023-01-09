#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import 
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8) #adjust the config. of the plots will create



# In[23]:


#read in te data
df= pd.read_csv("movies1.csv")


# In[24]:


#VIew in the data
df.head()


# In[25]:


#seeing if there is any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{}%.'.format(col,pct_missing))


# In[26]:


#data types for ou colums 
df.dtypes


# In[39]:


#change data type of columns 
#df['budget'] = df['budget'].astype('int')
#df['gross'] = df['gross'].astype('int')


# In[31]:


#Create correct Year column
df['Year Correct']= df['released'].astype(str).str[:4]
df


# In[43]:


df = df.sort_values(by=['gross'],inplace= False,ascending = False )


# In[35]:


pd.set_option('display.max_rows',None)


# In[40]:


#drop any duplicates 
df.drop_duplicates()


# In[41]:


#Budget high correlation
#company high correlation 


# In[46]:


#Scatter plot with budget vs gross

plt.scatter(x=df['budget'],y=df['gross'])
plt.title('Budget Vs Gross Earnings')
plt.xlabel('Gross Margins')
plt.ylabel('Budget For Film')
plt.show


# In[44]:


df.head()


# In[49]:


#plot budget vs gross using seaborn

sns.regplot(x='budget',y='gross',data=df,scatter_kws={'color':"green"},line_kws={"color":"blue"})


# In[50]:


#lookng in correlations


# In[52]:


df.corr(method='pearson')


# In[55]:


correlation_matrix=df.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matic for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features') 
plt.show()


# In[56]:


#looks at company 
df.head()


# In[61]:


df_numerized= df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype=="object"):
        df_numerized[col_name]=df_numerized[col_name].astype('category')
        df_numerized[col_name]=df_numerized[col_name].cat.codes
        
df_numerized


# In[66]:


#View in old data and nemerized data
old_file = pd.read_csv("movies1.csv")

old_file 


# In[63]:


correlation_matrix=df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matic for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features') 
plt.show()


# In[68]:


correlation_mat=df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[69]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[72]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#Votes and budget have the highest correlation to gross earnings

#company has low correlationn

