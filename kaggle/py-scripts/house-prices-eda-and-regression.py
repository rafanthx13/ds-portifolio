
# coding: utf-8

# <h1 style="text-align: center;">House Prices: EDA and Regression</h1>
# 
# <h3 align="center">Made by 🚀 <a href="https://www.kaggle.com/rafanthx13"> Rafael Morais de Assis</a></h3>
# 
# <img src="https://www.laoistoday.ie/wp-content/uploads/2018/09/house-prices-up2-1.jpg" />
# 
# Created: 2020-08-24; 
# 
# Last updated: 2020-08-24;
# 
# in progres.....
# 
# **Next Ideas**
# + Change Pre-Processing, Test other values in subimition, tests other combinations of models and weight
# 
# ## References
# 
# + https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# + https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# + https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition
# + https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1

# ## Abstract
# 
# Objetivo
# + Com base em 79 features, vamos tentar prever o preço de casas (regressão)
# 
# Conhecimentos aprendidos neste notebook:
# + Distribuiçâo Normal: skewness, kurtoise, boxcox, teste de normalidade
# + Diversars técnicas de regressão
# + Feature engineering: Diversass ideias e forma para este dataset
# + Data Missing e como lidar
# + Análise exploratória para ver tendencias e retirar outiliers
# 
# Como foi feito
# + Cross Validation: Using 12-fold cross-validation
# + Models: On each run of cross-validation I fit 7 models (ridge, svr, gradient boosting, random forest, xgboost, lightgbm regressors)
# + Stacking: In addition, I trained a meta StackingCVRegressor optimized using xgboost
# + Blending: All models trained will overfit the training data to varying degrees. Therefore, to make final predictions, I blended their predictions together to get more robust predictions.

# ## Kaggle Description
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 
# ### Competition Desrciption
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# ### The Goal
# 
# Each row in the dataset describes the characteristics of a house.
# 
# Our goal is to predict the SalePrice, given these features.
# 
# Our models are evaluated on the Root-Mean-Squared-Error (RMSE) between the log of the SalePrice predicted by our model, and the log of the actual SalePrice. Converting RMSE errors to a log scale ensures that errors in predicting expensive houses and cheap houses will affect our score equally.
# 
# ### File Description
# 
# 
# + `train.csv` - the training set
# + `test.csv` - the test set
# + `data_description.txt` - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
# + `sample_submission.csv` - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms
# 
# ### DataSet Description
# 
# | Column                                                                                                          | Type/Values | \|\|\| | Column                                                                 | Type/Values |
# |-----------------------------------------------------------------------------------------------------------------|-------------|--------|------------------------------------------------------------------------|-------------|
# | SalePrice - the property's sale price in dollars.<br>This is the target variable that you're trying to predict. |             | \|\|\| | HeatingQC: Heating quality and condition                               |             |
# | MSSubClass: The building class                                                                                  |             | \|\|\| | CentralAir: Central air conditioning                                   |             |
# | MSZoning: The general zoning classification                                                                     |             | \|\|\| | Electrical: Electrical system                                          |             |
# | LotFrontage: Linear feet of street connected to property                                                        |             | \|\|\| | 1stFlrSF: First Floor square feet                                      |             |
# | LotArea: Lot size in square feet                                                                                |             | \|\|\| | 2ndFlrSF: Second floor square feet                                     |             |
# | Street: Type of road access                                                                                     |             | \|\|\| | LowQualFinSF: Low quality finished square feet (all floors)            |             |
# | Alley: Type of alley access                                                                                     |             | \|\|\| | GrLivArea: Above grade (ground) living area square feet                |             |
# | LotShape: General shape of property                                                                             |             | \|\|\| | BsmtFullBath: Basement full bathrooms                                  |             |
# | LandContour: Flatness of the property                                                                           |             | \|\|\| | BsmtHalfBath: Basement half bathrooms                                  |             |
# | Utilities: Type of utilities available                                                                          |             | \|\|\| | FullBath: Full bathrooms above grade                                   |             |
# | LotConfig: Lot configuration                                                                                    |             | \|\|\| | HalfBath: Half baths above grade                                       |             |
# | LandSlope: Slope of property                                                                                    |             | \|\|\| | Bedroom: Number of bedrooms above basement level                       |             |
# | Neighborhood: Physical locations within <br>Ames city limits                                                    |             | \|\|\| | Kitchen: Number of kitchens                                            |             |
# | Condition1: Proximity to main road or railroad                                                                  |             | \|\|\| | KitchenQual: Kitchen quality                                           |             |
# | Condition2: Proximity to main road or railroad <br>(if a second is present)                                     |             | \|\|\| | TotRmsAbvGrd: Total rooms above grade <br>(does not include bathrooms) |             |
# | BldgType: Type of dwelling                                                                                      |             | \|\|\| | Functional: Home functionality rating                                  |             |
# | HouseStyle: Style of dwelling                                                                                   |             | \|\|\| | Fireplaces: Number of fireplaces                                       |             |
# | OverallQual: Overall material and finish quality                                                                |             | \|\|\| | FireplaceQu: Fireplace quality                                         |             |
# | OverallCond: Overall condition rating                                                                           |             | \|\|\| | GarageType: Garage location                                            |             |
# | YearBuilt: Original construction date                                                                           |             | \|\|\| | GarageYrBlt: Year garage was built                                     |             |
# | YearRemodAdd: Remodel date                                                                                      |             | \|\|\| | GarageFinish: Interior finish of the garage                            |             |
# | RoofStyle: Type of roof                                                                                         |             | \|\|\| | GarageCars: Size of garage in car capacity                             |             |
# | RoofMatl: Roof material                                                                                         |             | \|\|\| | GarageArea: Size of garage in square feet                              |             |
# | Exterior1st: Exterior covering on house                                                                         |             | \|\|\| | GarageQual: Garage quality                                             |             |
# | Exterior2nd: Exterior covering on house <br>(if more than one material)                                         |             | \|\|\| | GarageCond: Garage condition                                           |             |
# | MasVnrType: Masonry veneer type                                                                                 |             | \|\|\| | PavedDrive: Paved driveway                                             |             |
# | MasVnrArea: Masonry veneer area in square feet                                                                  |             | \|\|\| | WoodDeckSF: Wood deck area in square feet                              |             |
# | ExterQual: Exterior material quality                                                                            |             | \|\|\| | OpenPorchSF: Open porch area in square feet                            |             |
# | ExterCond: Present condition of <br>the material on the exterior                                                |             | \|\|\| | EnclosedPorch: Enclosed porch area in square feet                      |             |
# | Foundation: Type of foundation                                                                                  |             | \|\|\| | 3SsnPorch: Three season porch area in square feet                      |             |
# | BsmtQual: Height of the basement                                                                                |             | \|\|\| | ScreenPorch: Screen porch area in square feet                          |             |
# | BsmtCond: General condition of the basement                                                                     |             | \|\|\| | PoolArea: Pool area in square feet                                     |             |
# | BsmtExposure: Walkout or garden level basement walls                                                            |             | \|\|\| | PoolQC: Pool quality                                                   |             |
# | BsmtFinType1: Quality of basement finished area                                                                 |             | \|\|\| | Fence: Fence quality                                                   |             |
# | BsmtFinSF1: Type 1 finished square feet                                                                         |             | \|\|\| | MiscFeature: Miscellaneous feature not <br>covered in other categories |             |
# | BsmtFinType2: Quality of second <br>finished area (if present)                                                  |             | \|\|\| | MiscVal: $Value of miscellaneous feature                               |             |
# | BsmtFinSF2: Type 2 finished square feet                                                                         |             | \|\|\| | MoSold: Month Sold                                                     |             |
# | BsmtUnfSF: Unfinished square feet of basement area                                                              |             | \|\|\| | YrSold: Year Sold                                                      |             |
# | TotalBsmtSF: Total square feet of basement area                                                                 |             | \|\|\| | SaleType: Type of sale                                                 |             |
# | Heating: Type of heating                                                                                        |             | \|\|\| | SaleCondition: Condition of sale                                       |             |

# ## Table Of Contents (TOC) <a id="top"></a>
# 
# + [Import Libs and DataSet](#index01) 
# + [Snippets](#index02)
# + [EDA on 'SalePrice': the target to be predicted](#index03)
#   - [Top Correlation with 'SalePrice'](#index04)
#   - [Outiliers to 'SalePrice' to top corr features](#index05)
#   - [Remove Outiliers](#index06)
#   - [See 'SalePrice' as normal distribution](#index07)
#   - [Transform 'SalePrice' in a 'correct' normal distribution](#index08)
# + [Data Cleaning](#index09)
#   - [Join Train and Test Datasets to cleaning](#index10)
#   - [Missing Data](#index11)
#   - [Fix skewness in features to be normal distributions](#index12)
# + [Feature engineering](#index13)
#   - [Create New Features](#index14)
#   - [Encoded Categorical Features](#index15)
# + [Recreate Train nad Test DataSets](#index16)
# + [Developing models](#index17)
#   - [Evaluate models with CrossValidation](#index18)
#   - [Fit Models](#index19)
#   - [Join Models in Blend Model](#index32)
#   - [Evaluate Blend, Stack and all others models](#index33)
# + [Submit Prediction](#index20)
# 

# ## Import Libs and DataSet <a id='index01'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score

import warnings
import time
warnings.filterwarnings("ignore")

# statistics
from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax #for some statistics
from scipy.special import boxcox1p


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.3f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)


# In[3]:


file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
df_train = pd.read_csv(file_path)
print("Train DataSet = {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]))
print("Columns:", df_train.columns.tolist())
df_train.head()


# In[4]:


file_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
df_test = pd.read_csv(file_path)
print("Test DataSet = {} rows and {} columns".format(df_test.shape[0], df_test.shape[1]))
print("Columns:", df_test.columns.tolist())
df_test.head()


# In[5]:


quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']
print("Qualitative Variables: (Numerics)", "\n\n=>", qualitative,
      "\n\nQuantitative Variable: (Strings)\n=>", quantitative)


# ## Snippets <a id='index02'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 

# In[6]:


def eda_numerical_feat(series, title="", with_label=True, number_format=""):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
    print(series.describe())
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    if(with_label):
        describe = series.describe()
        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
              'Q3': describe.loc['75%']}
        if(number_format != ""):
            for k, v in labels.items():
                height = 0.3
                if(k == 'Q2'):
                    height = -0.3
                ax2.text(v, height, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
                         size=12, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=8, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()


# In[7]:


def plot_top_rank_correlation(my_df, column_target, top_rank=5):
    corr_matrix = my_df.corr()
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)

    ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
    ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 8}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
    cols_bot  = cols_bot.insert(0, column_target)
    cm = np.corrcoef(my_df[cols_bot].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 8}, yticklabels=cols_bot.values,
                     xticklabels=cols_bot.values, mask=mask, ax=ax2)
    
    plt.show()


# In[8]:


def test_normal_distribution(serie, series_name='series', thershold=0.4):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    f.suptitle('{} is a Normal Distribution?'.format(series_name), fontsize=18)
    ax1.set_title("Histogram to " + series_name)
    ax2.set_title("Q-Q-Plot to "+ series_name)
    
    # calculate normal distrib. to series
    mu, sigma = norm.fit(serie)
    print('Normal dist. (mu= {:,.2f} and sigma= {:,.2f} )'.format(mu, sigma))
    
    # skewness and kurtoise
    skewness = serie.skew()
    kurtoise = serie.kurt()
    print("Skewness: {:,.2f} | Kurtosis: {:,.2f}".format(skewness, kurtoise))
    # evaluate skeness
    # If skewness is less than −1 or greater than +1, the distribution is highly skewed.
    # If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.
    # If skewness is between −½ and +½, the distribution is approximately symmetric.
    pre_text = '\t=> '
    if(skewness < 0):
        text = pre_text + 'negatively skewed or left-skewed'
    else:
        text =  pre_text + 'positively skewed or right-skewed\n'
        text += pre_text + 'in case of positive skewness, log transformations usually works well.\n'
        text += pre_text + 'np.log(), np.log1(), boxcox1p()'
    if(skewness < -1 or skewness > 1):
        print("Evaluate skewness: highly skewed")
        print(text)
    if( (skewness <= -0.5 and skewness > -1) or (skewness >= 0.5 and skewness < 1)):
        print("Evaluate skewness: moderately skewed")
        print(text)
    if(skewness >= -0.5 and skewness <= 0.5):
        print('Evaluate skewness: approximately symmetric')
    # evaluate kurtoise
    #     Mesokurtic (Kurtoise next 3): This distribution has kurtosis statistic similar to that of the normal distribution.
    #         It means that the extreme values of the distribution are similar to that of a normal distribution characteristic. 
    #         This definition is used so that the standard normal distribution has a kurtosis of three.
    #     Leptokurtic (Kurtosis > 3): Distribution is longer, tails are fatter. 
    #         Peak is higher and sharper than Mesokurtic, which means that data are heavy-tailed or profusion of outliers.
    #         Outliers stretch the horizontal axis of the histogram graph, which makes the bulk of the data appear in a 
    #         narrow (“skinny”) vertical range, thereby giving the “skinniness” of a leptokurtic distribution.
    #     Platykurtic: (Kurtosis < 3): Distribution is shorter, tails are thinner than the normal distribution. The peak
    #         is lower and broader than Mesokurtic, which means that data are light-tailed or lack of outliers.
    #         The reason for this is because the extreme values are less than that of the normal distribution.
    print('evaluate kurtoise')
    if(kurtoise > 3 + thershold):
        print(pre_text + 'Leptokurtic: anormal: Peak is higher')
    elif(kurtoise < 3 - thershold):
        print(pre_text + 'Platykurtic: anormal: The peak is lower')
    else:
        print(pre_text + 'Mesokurtic: normal: the peack is normal')
    
    # shapiro-wilki test normality
    # If the P-Value of the Shapiro Wilk Test is larger than 0.05, we assume a normal distribution
    # If the P-Value of the Shapiro Wilk Test is smaller than 0.05, we do not assume a normal distribution
    #     print("Shapiro-Wiki Test: Is Normal Distribution? {}".format(stats.shapiro(serie)[1] < 0.01) )
    #     print(stats.shapiro(serie))

    
    # ax1 = histogram
    sns.distplot(serie , fit=norm, ax=ax1)
    ax1.legend(['Normal dist. ($\mu=$ {:,.2f} and $\sigma=$ {:,.2f} )'.format(mu, sigma)],
            loc='best')
    ax1.set_ylabel('Frequency')
    # ax2 = qq-plot
    stats.probplot(df_train['SalePrice'], plot=ax2)
    plt.show()
    


# In[119]:


def plot_model_score_regression(models_name_list, model_score_list, title=''):
    fig = plt.figure(figsize=(15, 6))
    ax = sns.pointplot( x = models_name_list, y = model_score_list, 
        markers=['o'], linestyles=['-'])
    for i, score in enumerate(model_score_list):
        ax.text(i, score + 0.002, '{:.6f}'.format(score),
                horizontalalignment='left', size='large', 
                color='black', weight='semibold')
    plt.ylabel('Score', size=20, labelpad=12)
    plt.xlabel('Model', size=20, labelpad=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.title(title, size=20)

    plt.show()


# ## EDA on 'SalePrice': the target to be predicted <a id='index03'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 
# 

# ### Top Correlation with 'SalePrice' <a id='index04'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[10]:


plot_top_rank_correlation(df_train, 'SalePrice', 7)


# OverallQual and GrLivArea tem as maiores correlações, depois GarageCars e GarageArea

# ### Outiliers to 'SalePrice' to top corr features <a id='index05'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[11]:


data = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)
data.plot.scatter(x='GarageArea', y='SalePrice', alpha=0.3, ylim=(0,800000));


# In[12]:


data = pd.concat([df_train['SalePrice'], df_train['GarageCars']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=df_train['GarageCars'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[13]:


data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.3, ylim=(0,800000));


# In[14]:


data = pd.concat([df_train['SalePrice'], df_train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0,800000));


# In[15]:


f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), ax=ax1);
ax1.set_title('SalesPrice x GrLivArea')

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), ax=ax2);
ax2.set_title('SalesPrice x TotalBsmtSF')


# In[16]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[17]:


data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=df_train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=45);


# consideraçoes

# In[18]:


print(df_train.shape[0])


# ### Remove Outiliers <a id='index06'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[19]:


rows_before = df_train.shape[0]
# Remove outliers
df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index, inplace=True)
df_train.drop(df_train[(df_train['GrLivArea']>4500) & (df_train['SalePrice']<300000)].index, inplace=True)
# eu adicinei, pois nao segue a reta  normal
df_train.drop(df_train[(df_train['LotArea']>100000)].index, inplace=True)
df_train.reset_index(drop=True, inplace=True)
rows_after = df_train.shape[0]
print("Qtd Row removed outiliers:", rows_before - df_train.shape[0])


# ### See 'SalePrice' as normal distribution <a id='index07'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[20]:


eda_numerical_feat(df_train['SalePrice'], "SalePrice", number_format='{:.0f}')

## color subtitulos


# In[21]:


test_normal_distribution(df_train['SalePrice'] , 'SalePrice')


# ### Transform 'SalePrice' in a 'correct' normal distribution <a id='index08'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[22]:


# log(1+x) transform
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])


# In[23]:


test_normal_distribution(df_train['SalePrice'] , 'SalePrice')


# ## Box Cow to target (A normal Distribution)
# 
# http://www.portalaction.com.br/analise-de-capacidade/411-transformacao-de-box-cox
# 
# Quando a distribuição normal não se adéqua aos dados, muitas vezes é útil aplicar a transformação de Box-Cox para obtermos a normalidade. Considerando X1, ..., Xn os dados originais, a transformação de Box-Cox consiste em encontrar um λ tal que os dados transformados Y1, ..., Yn se aproximem de uma distribuição normal. Esta transformação é dada por
# 
# Esse comportamento, posteriormente foi apresentado como a Curva de Gauss. Que mostrava que grande parte dos eventos ficam em torno de um valor médio, com uma certa variabilidade. Você sabe o que é uma curva de distribuição normal? Ou o que essa história que te contei tem haver com isso? Sabe qual a sua importância e para que serve? E como calcular?
# 
# Leia mais em: https://www.voitto.com.br/blog/artigo/distribuicao-normal
# 
# https://www.voitto.com.br/blog/artigo/distribuicao-normal

# ## Data Cleaning <a id='index09'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 
# ### Join Train and Test Datasets to cleaning <a id='index10'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[24]:


# Split features and labels
y_train = df_train['SalePrice'].reset_index(drop=True)
train_features = df_train.drop(['SalePrice'], axis=1)
test_features = df_test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
df_all = pd.concat([train_features, test_features]).reset_index(drop=True)
df_all.shape


# In[25]:


# ntrain = df_train.shape[0]
# ntest = df_test.shape[0]
# y_train = df_train.SalePrice.values
# df_all = pd.concat((df_train, df_test)).reset_index(drop=True)
# df_all.drop(['SalePrice'], axis=1, inplace=True)
# print("all_data size is : {}".format(df_all.shape))


# ## Missing Data <a id='index11'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[26]:


sns.heatmap(df_train.isnull(), cbar=False)


# In[27]:


def df_rating_missing_data(my_df):
    """Create DataFrame with Missing Rate
    """
    all_data_nan = (my_df.isnull().sum() / len(my_df)) * 100
    all_data_nan = all_data_nan.drop(all_data_nan[all_data_nan == 0].index).sort_values(ascending=False)[:30]
    return pd.DataFrame({'Missing Ratio' :all_data_nan})  


# In[28]:


df_missing_data = df_rating_missing_data(df_train)
df_missing_data


# In[29]:


f, ax = plt.subplots(figsize=(15, 4))
plt.xticks(rotation='90')
sns.barplot(x=df_missing_data.index, y=df_missing_data['Missing Ratio'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[30]:


# Some of the non-numeric predictors are stored as numbers; convert them into strings 
df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)
df_all['YrSold'] = df_all['YrSold'].astype(str)
df_all['MoSold'] = df_all['MoSold'].astype(str)


# In[31]:


# the data description states that NA refers to typical ('Typ') values
df_all['Functional'] = df_all['Functional'].fillna('Typ')
# Replace the missing values in each of the columns below with their mode
df_all['Electrical'] = df_all['Electrical'].fillna("SBrkr")
df_all['KitchenQual'] = df_all['KitchenQual'].fillna("TA")
df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])
df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
df_all['MSZoning'] = df_all.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# the data description stats that NA refers to "No Pool"
df_all["PoolQC"] = df_all["PoolQC"].fillna("None")
# Replacing the missing values with 0, since no garage = no cars in garage
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_all[col] = df_all[col].fillna(0)
# Replacing the missing values with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df_all[col] = df_all[col].fillna('None')
# NaN values for these categorical basement df_all, means there's no basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_all[col] = df_all[col].fillna('None')

# Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
df_all['LotFrontage'] = df_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# We have no particular intuition around how to fill in the rest of the categorical df_all
# So we replace their missing values with None
objects = []
for i in df_all.columns:
    if df_all[i].dtype == object:
        objects.append(i)
df_all.update(df_all[objects].fillna('None'))

# And we do the same thing for numerical df_all, but this time with 0s
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in df_all.columns:
    if df_all[i].dtype in numeric_dtypes:
        numeric.append(i)
df_all.update(df_all[numeric].fillna(0))


# In[32]:


df_rating_missing_data(df_all)


# ### Fix skewness in features to be normal distributions <a id='index12'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[33]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in df_all.columns:
    if df_all[i].dtype in numeric_dtypes:
        numeric.append(i)


# In[34]:


# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=df_all[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[35]:


# Find skewed numerical features
skew_features = df_all[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)


# In[36]:


# Normalize skewed features
for i in skew_index:
    df_all[i] = boxcox1p(df_all[i], boxcox_normmax(df_all[i] + 1))


# In[37]:


# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=df_all[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# ## Feature engineering <a id='index13'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 
# ### Create New Features <a id='index14'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 

# In[38]:


df_all['BsmtFinType1_Unf'] = 1*(df_all['BsmtFinType1'] == 'Unf')
df_all['HasWoodDeck'] = (df_all['WoodDeckSF'] == 0) * 1
df_all['HasOpenPorch'] = (df_all['OpenPorchSF'] == 0) * 1
df_all['HasEnclosedPorch'] = (df_all['EnclosedPorch'] == 0) * 1
df_all['Has3SsnPorch'] = (df_all['3SsnPorch'] == 0) * 1
df_all['HasScreenPorch'] = (df_all['ScreenPorch'] == 0) * 1
df_all['YearsSinceRemodel'] = df_all['YrSold'].astype(int) - df_all['YearRemodAdd'].astype(int)
df_all['Total_Home_Quality'] = df_all['OverallQual'] + df_all['OverallCond']
df_all = df_all.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['YrBltAndRemod'] = df_all['YearBuilt'] + df_all['YearRemodAdd']

df_all['Total_sqr_footage'] = (df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] +
                                 df_all['1stFlrSF'] + df_all['2ndFlrSF'])
df_all['Total_Bathrooms'] = (df_all['FullBath'] + (0.5 * df_all['HalfBath']) +
                               df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath']))
df_all['Total_porch_sf'] = (df_all['OpenPorchSF'] + df_all['3SsnPorch'] +
                              df_all['EnclosedPorch'] + df_all['ScreenPorch'] +
                              df_all['WoodDeckSF'])
df_all['TotalBsmtSF'] = df_all['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
df_all['2ndFlrSF'] = df_all['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
df_all['GarageArea'] = df_all['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
df_all['GarageCars'] = df_all['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
df_all['LotFrontage'] = df_all['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
df_all['MasVnrArea'] = df_all['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

df_all['haspool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['has2ndfloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasgarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasbsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['hasfireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[39]:


def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

df_all = logs(df_all, log_features)


# In[40]:


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']
df_all = squares(df_all, squared_features)


# ### Encoded Categorical Features <a id='index15'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[41]:


print('before', df_all.shape)
df_all = pd.get_dummies(df_all).reset_index(drop=True)
print('after encoded categorical features', df_all.shape)
# Remove any duplicated column names
df_all = df_all.loc[:,~df_all.columns.duplicated()]
print('after remove duplicate', df_all.shape)


# ## Recreate Train nad Test DataSets <a id='index16'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[42]:


X_train = df_all.iloc[:len(y_train), :]
X_test = df_all.iloc[len(y_train):, :]
X_train.shape, y_train.shape, X_test.shape


# ## Developing models <a id='index17'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# ### Evaluate models with CrossValidation <a id='index018'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 
# Cross valaidation serve para avaliar o mdoelo para novos dados. 
# 
# Diferente de t reinar e testar, queremos saber para o nosso modelo o quao bom ele é pra tetar em outros dados, pois sua estratgia é treinar/testar sobre dados diversos.
# 
# Exemplop:
# 
# Imagina que você faz um modello e elete tem 90% de acerto no conjutno de teste.
# 
# Depois, vocÊ aplica esse modelo em produçâo e tem resultados de 70%.
# 
# O que pode ter acontecido?
# 
# Talvez para os dados de testse ele se sai muito bem, mas para dados difernetes do de testse saia ruim (Overfittin).
# 
# ENtao usamos Cross validation com K-Fold: 
# + Dividmiso o conjunto de treino em K partes, 
# + usasmos a maior parte para trieno e uma pequena para tests
# + Obtemos resultados avaliandao treinamento e tests variandos ambos.
# + Assim fazemos a media e temos mais ideia de quais modelos sâo mais estáveis (vendo o desvio padrao) e quais tem melhores resultados na media (avalaindao a media)

# In[43]:


from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, scale
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LassoCV, BayesianRidge, LassoLarsIC
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# In[44]:


# Setup cross validation folds
kf = KFold(n_splits=4, random_state=42, shuffle=True)

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[53]:


# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# setup models    
lasso_alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

elastic_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
elastic_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# Lasso Regressor
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=lasso_alphas2,
                              random_state=42, cv=kf))
# Elastic Net Regressor
elasticnet = make_pipeline(RobustScaler(),  
                           ElasticNetCV(max_iter=1e7, alphas=elastic_alphas,
                                        cv=kf, l1_ratio=elastic_l1ratio))

# Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors = (xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor = xgboost,
                                use_features_in_secondary=True)


# In[56]:


regressor_models = {
    'LightGB': lightgbm, # 20s
    'XGBoost': xgboost, # 340s = 5min 40s
    'SVM_Regressor': svr, # 6s
    'Ridge': ridge, # 6s
    'RandomForest': rf, # 146s = 2min 20s
    'GradientBoosting': gbr, # 93s = 1min 30s
    # 'stack_gen': stack_gen, # Nâo tem como fazer, esse CV é para avaliar os outros, nao tem como aplicar o CV ao Stack
    ## ADD++
    'Lasso': lasso, # 15s
    'KernelRidge': KRR, # 1.88s
    'ElasticNet': elasticnet # 40s
}

scores = {}

## Cross Validation
t_start = time.time()

for model_name, model in regressor_models.items():
    print(model_name)
    t0 = time.time()
    score = cv_rmse(model)
    t1 = time.time()
    m, s = score.mean(), score.std()
    scores[model_name] = [m,s]
    print('\t=> mean {:.5f}, std: {:.5f}'.format(m, s))
    print("\t=> took {:,.3f} s".format(t1 - t0))
    
t_ending = time.time()
print('took', t_ending - t_start)


# In[61]:


plot_model_score_regression(list(scores.keys()),
                            [score for score, _ in scores.values()],
                            'Best Individual Models by Mean score (RMSLE)')


# In[60]:


plot_model_score_regression(list(scores.keys()), 
                            [score for _, score in scores.values()],
                            'Best Individual Models by std score (RMSLE)')


# ### Fit models <a id='index19'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[62]:


# Def Stack Model: Stack up all the models above, optimized using  ['xgboost'/'elasticnet']
stack_gen = StackingCVRegressor(regressors = (xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor = elasticnet,
                                use_features_in_secondary=True)

# train a model and show the time
def fit_a_model(model, model_name):
    t0 = time.time()
    if(model_name == 'Stack'):
        a_model = model.fit( np.array(X_train), np.array(y_train) )
    else:
        a_model =  model.fit( X_train, y_train )
    t1 = time.time()
    print("{} took {:,.3f} s".format(model_name, t1 - t0))
    return a_model

lgb_model   = fit_a_model(lightgbm, 'LightGB') # 3.7s
svr_model   = fit_a_model(svr, 'SVM_R') # 1.8s
ridge_model = fit_a_model(ridge, 'Ridge') # 1.8s
gbr_model   = fit_a_model(gbr, 'GradientBoost') # 30s
lasso_model = fit_a_model(lasso, 'Lasso') # 2.8s
kridg_model = fit_a_model(KRR, 'KernelRidge') # 1.2
elast_model = fit_a_model(elasticnet, 'ElasticNet') # 8s
# more time
rf_model    = fit_a_model(rf, 'RandomForest') # 51s
xgb_model   = fit_a_model(xgboost, 'XGboost') # 116s = 2min
stack_model = fit_a_model(stack_gen, 'Stack') # 1.087s = 18min


# ### Join Models in Blend Model <a id='index32'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[133]:


# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.10 * ridge_model.predict(X)) + #             (0.15 * kridg_model.predict(X)) + \
            (0.30 * gbr_model.predict(X)) + \
            (0.15 * elast_model.predict(X)) + \
            (0.15 * lgb_model.predict(X)) + \
#             (0.05 * rf_model.predict(X)) + \
            (0.30 * stack_model.predict(np.array(X))))


# LAST BEST
#     return ((0.10 * ridge_model.predict(X)) + \
#             (0.15 * svr_model.predict(X)) + \
#             (0.15 * gbr_model.predict(X)) + \
#             (0.15 * elast_model.predict(X)) + \
#             (0.15 * lgb_model.predict(X)) + \
# #             (0.05 * rf_model.predict(X)) + \
#             (0.30 * stack_model.predict(np.array(X))))

# Original
#     return ((0.10 * ridge_model.predict(X)) + \
#             (0.15 * svr_model.predict(X)) + \
#             (0.10 * gbr_model.predict(X)) + \
#             (0.15 * xgb_model.predict(X)) + \
#             (0.10 * lgb_model.predict(X)) + \
#             (0.05 * rf_model.predict(X)) + \
#             (0.35 * stack_model.predict(np.array(X))))

# Others Ideas: Use stack with others models

# Get final precitions from the blended model
blended_score = rmsle(y_train, blended_predictions(X_train))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data to Blend Model:')
print(blended_score)

# Mesmo tentando outras coisa, nehum resultado foi melhor do que o Original

# Original: 0.07972798382117793
# try1:     0.07926790876360021 (change xgb => elastic)
# try2:     0.07791727684371352 (remove rf_model and add 0.5 in lgb_model)
# try3:     0.0761370032672606  (remove 0.5 from stack and add in gbr_model)
# try4:     0.0723571304918255  (remove from pipe_line the secoond and the rf_model and put all (0.15 in gbd))


# In[135]:


# Evaluate Stack
stack_score = rmsle(y_train, stack_model.predict(np.array(X_train)))
scores['stack'] = (stack_score, 0)
print('RMSLE score on train data to Stack Model:\n\t=>', stack_score)


# ### Evaluate Blend, Stack and all others models <a id='index33'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

# In[120]:


plot_model_score_regression(list(scores.keys()), [score for score, _ in scores.values()])


# ## Submit Prediction <a id='index20'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>
# 

# In[136]:


# Read in sample_submission dataframe
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape


# In[137]:


# Append predictions from blended models
# We Apply expm1 beacuse, our model generate SaleṔrice conveterted by log1, this is a inverse operation
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test)))


# In[138]:


# Fix outleir predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression11.csv", index=False)


# In[140]:


# Scale predictions (BEST)
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression33.csv", index=False)

