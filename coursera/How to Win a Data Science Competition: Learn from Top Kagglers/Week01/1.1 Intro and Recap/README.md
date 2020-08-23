# 1.1 Intro and Recap (Week01)

## 1.1.1 Welcome to "How to win a data sscience competition"

Intro. Não há nada demais.

## 1.1.2 Compettion mechanics

**Outras Plataformas**

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/coursera/How to Win a Data Science Competition: Learn from Top Kagglers/img/1.1.1.01.png)

**Porque participar**

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/coursera/How to Win a Data Science Competition: Learn from Top Kagglers/img/1.1.1.02.png)

**Diferença de competiçôes de casos reais**

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/coursera/How to Win a Data Science Competition: Learn from Top Kagglers/img/1.1.1.03.png)

### Quiz - Compettion mechanics

**Pergunta 1**
Which of these methods are used by organizers to prevent overfitting to the test set using scores from the leaderboard?

+ [X]Splitting test for public/private
- The submissions are scored on public part of the test set while private score is only revealed when competition ends, thus it is not possible to overfit to private part

+ [X] Limiting number of submissions per day
  - This also makes "leaderboard probing" much harder

**Pergunta 2**
What are the main reasons for participating in data science competitions? Check all that apply.

+ [X] To have an opportunity for networking with other participants
  - Yes, the people on forums are very active and helpful.

+ [X] To learn basics of data science
  - Yes! For example there are plenty of public datasets on kaggle.com with a huge community of people who is also learning.

**Pergunta 3**
Imagine that you are participating in an image recognition competition and you want to use some publicly available set of images. Are you allowed to do this?


+ [X] You need to check the competition rules
  - If you have any doubt about competition mechanics and limitations -- check the rules or ask organizers on forums

**Pergunta 4**
Which of these things we should take care about during competitions? Check all that apply.

+ [X] Target metric values
  - This one is used for leaderboard ranking

## 1.1.3 Recap of main ML Algorithms

**Linear models**

Exemplos: SVC

Buscar ser uma especie de regressão linear, separa os dados em dois subconjuntos por uma linha

**Tree**

Exemplos: Decision Tree, Random Forest, GBTD (Gradient Bostet tree decision)

A ideia da arvore de decisao é como se fosse várias retas, uma  para cada feautres. acontece que sempre tem que ser várias retas, entao, se houver um caso em que so precisar de uma, haver X retas para tentar reproduzrir essa uma que um simples modelo lienar poderia reporduzir

Prox e Contras
+ Bom para dados tabulares (numericos)
+ nao captura bem dependenicas lineares entre as variaveis

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/coursera/How to Win a Data Science Competition: Learn from Top Kagglers/img/1.1.1.04.png)

**NN**

TensorFlow, Keras, PyTorch

O instrutor em especial gosta do pytorch por ser mais fácil de desenhhar redes complexas.

**Teorema do Almoço Gratis**

NÃO HÁ UM ALGORITMO QUE RESOLVA TUDO.

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/coursera/How to Win a Data Science Competition: Learn from Top Kagglers/img/1.1.1.05.png)

Nâo há metodo que perfoma bem em todas as tarefas

**As ferremanteas mais poderossa costumam ser**

Implementações de GBDT
+ [xgboost](https://xgboost.readthedocs.io/en/latest/get_started.html)
+ [LightGBM](https://github.com/Microsoft/LightGBM)

e redes neurais

## 1.1.4 Software/Hardware requirements

Amazon AWS caso precisar de mais poder computacional


