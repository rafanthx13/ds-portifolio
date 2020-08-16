# Como fazer Kaggle Kernels Templates

## ORDEM

+ Titulo
  - date created, last update, languages, author
+ Kaggle Descripiton
+ Brief Summary (data-length)
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

## Titulo

<div style="text-align: center;">

\# Air Pollution in Seoul: EDA with visualization by maps 🗺

<h3 align="center">Made by 🚀 <a href="https://www.kaggle.com/rafanthx13"> Rafael Morais de Assis</a></h3>

</div><br>

**Language:** English (🇺🇸) and Portuguese (🇧🇷)

Created: 2020-08-14; (14/08/2020)

Last updated: 2020-08-14; (14/08/2020)


## Ter em cada subtitulo

<a id="top"></a>

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white; margin-left: 20px;" data-toggle="popover">Go to TOC</a>

<span style='font-size: 15pt'>Analyse dangerous measurements of each pollutant</span>

## Summary of the data

We have 6 data files in University Rankings dataset, as following:

cwurData.csv (2200 rows, 14 columns)
education_expenditure_supplementary_data.csv (333 rows, 9 columns)
educational_attainment_supplementary_data.csv (79055 rows, 29 columns)
school_and_country_table.csv (818 rows, 2 columns)
shanghaiData.csv (4897 rows, 11 columns)
timesData.csv (2603 rows, 14 columns)

## Areas

### Conclusions

### Feedback
This Kernel is still under development. I would highly appreciate your feedback for improvement and, of course, if you like it, please upvote it!

# OTHERS

🇺🇸

🇧🇷

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

## A vírgula e os números em inglês

Tenha bastante atenção quando for usar as vírgulas e os pontos nos números em inglês! O motivo principal é o fato de usarmos, em português, a vírgula para separar as casas decimais mas, em inglês, utiliza-se o ponto.

PORTUGUÊS = $2.550,00, 1,12%, 18,5km, 2,2 milhões etc.
ENGLISH = $2,550.00, 1.12% ,18.5km, 2.2 million etc.
Até aqui não há grandes problemas, mas a confusão pode ser bem grande quando são usadas três casas decimais. Observe o seguinte exemplo: ao escrevermos “2,354 kg” em português, estamos nos referindo a um peso de pouco mais de dois quilos. Em inglês, são mais de duas toneladas!

About 1.2 million people live in crowded refugee camps in the West Bank, Gaza and countries that neighbor Israel. (CNN)
Cerca de 1,2 milhão de pessoas vivem em campos de refugiados superlotados na Cisjordânia, em Gaza e em países vizinhos de Israel.

