# EDA

## Passo a Passo EDA

**Introduction**
+ `df.info()`
  - Descobrir o tamanho em memória, linhas, colunas e tipos dessas colunas
+ `df.describe().T` 
  - O T para visualizar melhor
+ Renomear features para ao formato snake_case (tudo minusculo e em ingles)
+ Entender cada Feature (gerar md com seus nomes e descrição)
+ Lidar com missing values
+ converter tipos das columnas para os tipos corretos (o tipo object é para string, se nao for string e tiver isso, muda agora

**Missign/Null/Corrupted Values**
[link](https://dev.to/tomoyukiaota/visualizing-the-patterns-of-missing-value-occurrence-with-python-46dj)

````python
sns.heatmap(df.isnull(), cbar=False)
import missingno as msno
# matrix: more easy see missign data in rows
msno.matrix(df)
# bar: show quantity
msno.bar(df)
# correlation between null values
msno.heatmap(df)
# count nan rows
df.isnull().sum().max()
````

+ Saber a quantidade de missign value
  - df.isnull().sum()
  - len(Series) - Series.count()
+ replace missing values
  - df['feat'].fillna(df['feat'].median(), inplace = True)

**REmover certas linhas especificas**
# Remove outliers
df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index, inplace=True)

**EM PROBLEMAS DE CLASSIFICAÇÂO, É BOM FICAR LIGADO NO DESBALANCEAMENTO: Se for, a divisâo da base entre treino e testse também deve está balanceado**

**PREPROCESSAMENTO É FEITO ANTES DE DIVIDIR A BASE**

**ANTES DE COMEÇAR A MODELAR, FAÇA UMA CORRELAÇÂO: Altere as variaveis para numericas para conseguir fazer**

**Check Duplicates**

\# Toda a linha
df.duplicated(subset=None, keep='first').sum()

\# Repetiçâo por feature
df.duplicated(subset=feature, keep='first').sum()

**Variáveis Numéricas**
+ BoxPlot
  - sns.boxplot(x=df['feat'], showfliers=True)
+ statistics
  - df['feat'].describe()

**Variáveis Categóricas**
+ Snipeets eda\_g\_categorical



## Some Tasks


**pandas_profiling in kaggle**

````python
import pandas_profiling as pdp
profile_X_train = pdp.ProfileReport(X_train)
profile_X_train 
````

**Rename columns name**

````python
df.rename(
    columns={
        "DATA INICIAL": "start_date",
        "DATA FINAL": "end_date",
        "REGIÃO": "region",
        "ESTADO": "state",
        "PRODUTO": "fuel",
        "COEF DE VARIAÇÃO REVENDA": "coef_price"
    },
    inplace=True
)
````

**Delete column**

````python
df.drop(['B', 'C'], axis=1) # axis 1 = column | axis 0 = row
````

**Map Values**

````python
regions = {"SUL":"SOUTH", "SUDESTE":"SOUTHEAST", "CENTRO OESTE":"MIDWEST"}
gp["region"] = gp["region"].map(regions)
````

**Iterar Dictionary iterar**

```python
for value in dict.values():
	print(value)

for key in dict.keys():
	print(key)

for key, value in dict.items():
	print(key, value)
```

**Iterar row por row de um dataFrame**

```
for index, row in df.iterrows():
     print(index, row['inner_vlan'], row['outer_vlan'])
```

#### Formatação da Saída em Float: Sem isso sai como e^x

`pd.options.display.float_format = '{:,.2f}'.format`

+ para voltar ao normal

`pd.reset_option("display.float_format")`

Evitar notaçâo cientifica (e^x)
ROW:
`pd.set_option('display.float_format', lambda x: '%.2f' % x)`

**Ler valores em notaçâo científica**

O "e-01" indica o 10^e (no cass e = -1)

9.9229e11  = 992290000000
9.9229e2   = 992.29
1.0000e+00 = 1
9.9229e-01 = 0.99229
3.6789e-02 = 0.036789
1.0005e-06 = 0.0000010005

**Não mostrar nada desnecessário**

````python
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
````

## Zoom HeartMap most important correlation

````python
k = 10 #number of variables for heatmap
# get the 'k' largest correlations values columns with 'SalePrice'
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# The heartmap is order_by correlation value with 'Sale Price' in row of 'Sale Price'
````

#### Seaborn e printar dados faltantes em gráfico

````python
# set up aesthetic design
plt.style.use('seaborn')
sns.set_style('whitegrid')

# create NA plot for train data
plt.subplots(0,0,figsize = (15,3)) # positioning for 1st plot
train.isnull().mean().sort_values(ascending = False).plot.bar(color = 'blue')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.title('Missing values average per columns in TRAIN data', fontsize = 20)
plt.show()
````
