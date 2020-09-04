# Como fazer Kaggle Kernels Templates

## Titulo

<h1 align="center"> User Cars: EDA and Regression </h1>

<img src="https://mystrongad.com/MTS_MillerToyota/MTS_Interactive/Used/Used-Car-Toyota.png" width="50%" />

Created: 2020-09-01

Last updated: 2020-09-01

Kaggle Kernel made by 🚀 <a href="https://www.kaggle.com/rafanthx13"> Rafael Morais de Assis</a>


## Kaggle Description

````
/## Kaggle Description

/### Data Description

/### The Goal

/### File Description

/### DataSet Description
````

## TOC

````
\## Table Of Content (TOC) <a id="top"></a>
````

## Libs and DataSet

````python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import time
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

\# Configs
pd.options.display.float_format = '{:,.3f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
````

## 



## ORDEM

+ Titulo
  - date created, last update, languages, author
+ Kaggle Descripiton
+ Abstract
  - Para ser mais entendível é necessário, eu achei, ter um asbtract como o de um artigo. Ele terá: descriçâo do dataset, o significado de cada linha (nao é features, é row mesmo). O que foivisto, como foi resolvido, métrico, cuirosidesde do processo e por fim a pontuaçâo final e como foi feita
+ Brief Summary (data-length) of dataset
  - em formato de tabela markdown
  - Se for muito grande, nao colocar
+ TableOfContent

---

**Conclusion**

🇺🇸

🇧🇷

-- script

https://medium.com/@researchplex/the-easiest-way-to-convert-jupyter-ipynb-to-python-py-912e39f16917

convert .ipynb to .html

ipython nbconvert — to script abc.ipynb 

convert .ipynb to .py

ipython nbconvert us-police-shooting-eda-with-maps-visualisation.ipynb --to python


## Ter em cada subtitulo

MEDICAL COST

+ [Import Libs and DataSet](#index01) 
+ [Snippets](#index02)
+ [EDA](#index03)
  - [Each feature individually](#index03)
  - [Each Feauture with 'charges'](#index04)
  - [Analyze feature crossover](#index05)
  - [Conclusions of EDA](#index06)
+ [Pre-Processing](#index07)
+ [Correlation](#index08)
+ [Split in Train and Test](#index09)
+ [Develop Models](#index10)
  - [Cross Validation](#index11)
  - [Fit Models](#index12)
  - [Test Models](#index13)
  - [Bests Models](#index14)
+ [Feature Importance](#index15)
+ [Hyperparameter Tuning Best Model](#index16)
+ [Evaluate Best Model to Regression](#index20)
+ [Conclusion](#index25)

+ [Import Libs and DataSet](#index01) 
+ [Snippets](#index02)
+ [Data Cleaning](#index03)
+ [EDA](#index04)
  - [Each feature Individually](#index04)
  - [Target by Features](#index05)
  - [Target by cross Features](#index06)
  - [EDA conclusions](#index50)
+ [Pre-Processing](#index07)
+ [Correlations](#index08)
+ [Split in train and Test](#index09)
+ [Develop Models](#index10)
  - [Prepare ML Models and Training](#index33)
  - [Cross Validation](#index11)
  - [Fit Models](#index12)
  - [Test Models](#index13)
  - [Bests Models](#index14)
+ [Feature Importance](#index15)
+ [Evaluate Best Model to Regression](#index20)
+ [Conclusion](#index25)

----

Target by Feaute
+ Year: The bigger the year tends to be the higher the price
+ Name: Some Names values more than others
+ Location: Coimbatore and Bangalore has more than others
+ Fuel_Type: Diesel has more price than PEtrol, and the others have few examples to check better
+ Kilometers_Driven: Few Influence
+ Milege: Few Influence
+ Engine: Linear Infleunce
+ Power: Linear INfluence
+ Seats: Has2 places has more mean than others

Target by cross Features
+ Transmission: Tem influencia em Power, a parrtir de Power 200 so ha tramnissao automatica e tem os maiores preços. Ocorre de forma parecida com Engine. Analsisando 4 features nuemricas (Engine, Power, Mileage, Kilometres_Drive) vemos que ser transmisaao automatiac da de acrra um grande preço



-----

<a id="top"></a>

<a id='index02'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

<span style='font-size: 15pt'>Analyse_dangerous_measurements_of_each_pollutant</span>

## Summary of the data

We have 6 data files in University Rankings dataset, as following:

cwurData.csv (2200 rows, 14 columns)
education_expenditure_supplementary_data.csv (333 rows, 9 columns)
educational_attainment_supplementary_data.csv (79055 rows, 29 columns)
school_and_country_table.csv (818 rows, 2 columns)
shanghaiData.csv (4897 rows, 11 columns)
timesData.csv (2603 rows, 14 columns)

### Conclusions


------------

chart: 'charge' x 'smoke'

Como em 'charge x smoke' o fato de fumar é bem importnate, a distribuiçâo de charges é claramente diferente entre um fumante e um não fumante. A maior parte dos fumantes tem encargos muito maiores que os nâo fumantes

chart: charge by age

Quanto maior a idade maior o preço

chart: charges by bmi

Quanto maior o bmi maior é a tendencia de se ter grandes valores, apesar disso só o bmi nâo explicar grandes custos

chart: charge x age with others features

Em 'charges by age and smoke' podemo perceber nitidamente 3 classes. 
+ classe 1, menor gastos, são os não fumantes
+ classe 2, gastos medianos, fumantes e não fumantes
+ classe 3, maiores gastos, fumantes
Depois olhando para 'charges by age and weight_condition' temos que em grande maioria essa terceira classe é das pessoas obesas (alto bmi)

chart: 'charges by bmi with others features'

Em 'charges by bmi and age_cat' nos mostra a peça que falta, junto com 'charges by bmi and smoke'.

Somente olhando os dados dapra fazer mental,emtne uma árvore de decisao são fumo, idade e bmi.

+ Se fuma tera mais gastos que os não fulmantes (boa parte da população), gastos acima de 15,000
  - Analisa-se o BMI, se não for obseo, fica num grupo entre 18,000 e 30,000, se obeso acima de 35,000
  - Para cada um desses dois grupos, quanto maior a idade, mais caro fica

+ Se não fuma gastos abaixo de 15,000
  - Para os nao fulmenates o segundo critério seria a idade, quanto mais velho maior o gasto
  - Nisso o bmi não influencia muito. Apesar disso alguns com peso normal ou acima (NormalWeight, Overweight or Obese) podem cair no custo de serem tão caro quanto fumante, principlamente obesos

sexo, children e region influenciam bem pouco, isso também será visto na parte de correlações


--------------

### Feedback
This Kernel is still under development. I would highly appreciate your feedback for improvement and, of course, if you like it, please upvote it!

# OTHERS

🇺🇸
, 🇧🇷

## Random-Title <a id ='index01'></a>

## Table of Contents

Snippets <a id='index02'></a> <a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

+ [Import Libs and DataSet](#index01) 
+ [Snippets](#index02)
+ [Understand DataSet](#index03)
+ [Distribution of *Time* and *Amount*](#index04)
+ [Scalling *Time* and *Amount*](#index05)
+ [Split DataSet in Test and Train](#index06)
+ [Random Under-Sampling to correct unbalanced](#index07)
  - [Make Under-Sampling](#index08)
  - [View correlation on balanced dataset](#index09)
  - [Show correlation with *Class* in BoxsPlots](#index10)
  - [Remove Outiliers](#index11)
+ [Dimensionality Reduction and Clustering](#index12)
+ [Train on UnderSampling](#index13)
+ [Test in Original DataFrame Unbalanced](#index14)
+ [Oversampling with SMOTE](#index15)
  - [Create DataSet balanced with SMOTE](#index21)
  - [Test model in UnderSampling DataSet](#index20)
+ [Neural Network](#index16)
  - [UnderSampling - Random](#index17)
  - [OverSampling - SMOTE](#index18)


