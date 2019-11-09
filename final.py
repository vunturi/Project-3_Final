#!/usr/bin/env python
# coding: utf-8

# In[36]:


from sklearn import tree
#import os 
import pandas as pd

#data = pd.read_csv('Multiple Cause of Death, 1999-2017.txt',sep=" " header = None)

#data.columns = ["Notes","Census Region","Census Region Code","State","State Code","Gender","Gender Code","Age Groups","TenYear Age Groups Code",\
                #"Year","Year Code"]
#df = pd.read_csv('death rate west verginia.csv')

#df_Death = pd.read_csv("Multiple_Cause_of_Death_WV_1999-2017_clean.csv")
df_Pharma =pd.read_csv("WVPharmData.csv")


# In[37]:





#df_Pharma.to_string(columns = ["TRANSACTION_DATE"])
df_Pharma["transaction year"]= df_Pharma["TRANSACTION_DATE"].astype(str)
# slicing till 2nd last element 
#df_Pharma["transaction year"]= df_Pharma["transaction year"].str.slice(3, -1, 1) 
df_Pharma["transaction year"] = df_Pharma["transaction year"].apply(lambda x: x[-4:])

df_Pharma.rename(columns = {"BUYER_COUNTY":"County"}, inplace = True)
df_Pharma["County"]=df_Pharma["County"].astype(str)
#df_Pharma["County"].str.title()
df_Pharma["County"] =df_Pharma["County"].str.lower() 
df_Pharma.head()


# In[38]:


df_Pharma.sort_values(by=['QUANTITY'])


# In[39]:


df_Pharma.QUANTITY.unique()


# In[40]:



len(df_Pharma.County.unique())


# In[41]:


#target = df["Outcome"]
#target_names = ["negative", "positive"]
#df_Death["Year Code"]= df_Death["Year Code"].astype(str)
#df_Death["Year Code"]=df_Death["Year Code"].apply(lambda x: x[:4])
#df_Death["Deaths"]= df_Death["Deaths"].astype(str)
#df_Death["Deaths"]=df_Death["Deaths"].apply(lambda x: x[:2])
#df_Death["Population"]= df_Death["Population"].astype(str)
#df_Death["Population"]=df_Death["Population"].apply(lambda x: x[:4])
# df_Death["County"]=df_Death["County"].str.lower()
# df_Death.show()


# In[42]:


dfNEW = pd.read_table('input3.txt')
dfNEW.head()


# In[43]:


dfCounties= dfNEW[['County', 'Year', 'Population', 'Deaths']]
dfCounties.head()


# In[44]:


dfCounties['county']=dfCounties['County'].astype(str)


# In[45]:


dfCounties['county_new'] = dfCounties['county'].str.split(' ').str[0]


# In[46]:


dfCounties["countyF"] =dfCounties["county_new"].str.lower()
dfCounties.head()


# In[47]:


dfCounties=dfCounties.dropna()


# In[48]:


len(dfCounties)


# In[49]:


dfCounties=dfCounties[['countyF', 'Year', 'Population', 'Deaths']]
dfCounties.rename(columns = {"countyF":"County"}, inplace = True)
dfCounties.head()


# In[50]:


dfCounties['New_Year']=dfCounties['Year'].astype(str)
dfCounties.head()


# In[51]:


dfCounties['Year_new'] = dfCounties['New_Year'].str.split('.').str[0]
dfCounties.head()


# In[52]:


dfCounties=dfCounties[['County', 'Year_new', 'Population', 'Deaths']]
dfCounties.rename(columns = {"Year_new":"Year"}, inplace = True)
dfCounties.head()


# In[53]:


print(len(df_Pharma))


# In[54]:


merge_table = pd.merge(df_Pharma, dfCounties, how='left', left_on=['County','transaction year'], right_on = ['County','Year'])
merge_table

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
D=merge_table["Population"].values
kmeans.fit(D.reshape(-1,1))

# starting from here we need to figure out the machine learning model


# In[55]:


D=merge_table["Population"].values


# In[56]:


D.reshape(-1,1)


# In[60]:



predicted_clusters = KMeans.predict(D)
X = D["Population"]
Y = D["Deaths"]
feature_names = D.columns
predicted_clusters = KMeans.predict(D)
#D.head()


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)


# In[6]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[9]:


sorted(zip(rf.feature_importances_, feature_names), reverse=True)


# In[69]:


dfWVTOT = pd.read_table('ALLDEATHSWV.txt')
dfWVTOT.head()


# In[70]:


dfWVTot= dfWVTOT [['Year', 'Deaths', 'Population']]
len(dfWVTot)


# In[71]:


dfWVTot=dfWVTot.dropna()


# In[72]:


dfWVTot.head()


# In[73]:


dfWVDrug = pd.read_table('DRUGDEATHSWV.txt')
dfWVDrug.head()


# In[74]:


dfWVDrug= dfWVDrug [['Year', 'Deaths', 'Population']]


# In[75]:


dfWVDrug = dfWVDrug.dropna()


# In[76]:


len(dfWVDrug)


# In[77]:


dfWVDrug.rename(columns = {"Deaths":"Drug_Deaths"}, inplace = True)


# In[78]:


dfWVDrug.head()


# In[79]:


Totals_Merged= pd.merge(dfWVDrug,dfWVTot , on="Year")


# In[80]:


Totals_Merged


# In[82]:


Totals_Merged=Totals_Merged.drop('Population_x', axis=1)


# In[84]:


Totals_Merged.rename(columns = {"Population_y":"Population"}, inplace = True)


# In[85]:


Totals_Merged


# In[ ]:




