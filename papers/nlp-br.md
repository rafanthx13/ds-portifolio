---
layout: default
title: NLP PT-BR
description: Questões de DataScience e Machine Learning
---

## Índex

## Links
+ [https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo](https://br.bitdegree.org/tutoriais/data-science/#Dicas_gerais_e_resumo)
+ Ultra link: https://www.kaggle.com/rftexas/nlp-cheatsheet-master-nlp

## Glossário e Resumo de Termos

Alguns termos serão mantidos em inglês

+ NLP: Processamento de linguagem natural (PLN)
+ NLTK: [Natural Language Toolkit](https://www.nltk.org/) Pacote do Python para fazer NLP
+ stop words: Palavras sem sentido semântico, 

## Atividades Realizadas por NLP

+ Tradução
+ Reconhecimento de Fala/Voz
+ Correção Ortográfica
+ Sistema de Peguntas/Respostas
+ ChatBots
+  Análise de sentimentos em textos

## Tarefas de NLP

### Tokenization

Tokenização: separa o texto em partes. Em geral separa palavra-por-palavra



Pode ser definido como o processo de quebrar um pedaço de texto em partes menores, como frases e palavras. Essas partes menores são chamadas de tokens. Por exemplo, uma palavra é um token em uma frase, e uma frase é um token em um parágrafo.

 Os tokens, mencionados acima, são muito úteis para encontrar e compreender esses padrões. Podemos considerar a tokenização como a etapa base para outras receitas, como lematização e lematização

**Exemplo em NLTK**

````python
import nltk
from nltk.tokenize import word_tokenize
word_tokenize('Tutorialspoint.com provides high quality technical tutorials for free.')
# OUTPUT: ['Tutorialspoint.com', 'provides', 'high', 'quality', 'technical', 'tutorials', 'for', 'free', '.']
````

### Lemmatizing e Stemming

Escolhe-se uma ou outra para usar: São técnicas para derivar um nome para sua forma mais pura, a raiz

**Lemmatizing**

A técnica de lematização é como uma derivação. A saída que obteremos após a lematização é chamada de ‘lema’, que é uma raiz em vez de radical, a saída da lematização. Após a lematização, estaremos obtendo uma palavra válida que significa a mesma coisa.

O NLTK fornece a classe `WordNetLemmatizer`, que é um invólucro fino em torno do corpus do wordnet. Esta classe usa a função `morphy()` para a classe `WordNet` `CorpusReader` para encontrar um lema. Vamos entender com um exemplo -

**Exemplo com NLTK**

````python
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('books')
# OUTPUT: 'book'
````

**Stemming**

Stemming é uma técnica usada para extrair a forma básica das palavras removendo afixos delas. É como cortar os galhos de uma árvore até o caule. Por exemplo, a raiz das palavras eating, eats, eaten is eat.

Os mecanismos de pesquisa usam derivação para indexar as palavras. É por isso que, em vez de armazenar todas as formas de uma palavra, um mecanismo de pesquisa pode armazenar apenas as hastes. Dessa forma, a derivação reduz o tamanho do índice e aumenta a precisão da recuperação.

````python
import nltk
from nltk.stem import PorterStemmer
word_stemmer = PorterStemmer()
word_stemmer.stem('writing')
# OUTPUT: 'write'
````

### Annotations

+ Annotation ou Produção de Anotações é o processo de registrar elementos de composição do texto, como flexões e dependências
  + É o mais importante, pois vai organizar e documentar todo o documento para o restante do processo
  + Exemplo: https://udemy-images.s3.amazonaws.com/redactor/raw/2019-03-17_22-29-24-bece19bfb6ec8c609a4403909ad9404d.PNG
  + CoNLL-U é um formato, ou modelo padrão, para processamento de linguagem natural. O processo de fazer anotações é annotations

### Parts-of-Speech Tagging (POS) 

Em NLP, Parts-of-Speech Tagging, ou simplesmente POS, é o processo de adicionar tags a cada token, de acordo com o elemento encontrado, como substantivo, adjetivo etc.



Part of speech ou POS é uma função gramatical que explica como uma palavra específica é usada em uma frase. Existem oito classes gramaticais: substantivo, pronome, adjetivo, verbo, advérbio, preposição, conjunção, interjeição. A marcação de parte da fala é o processo de atribuição de uma etiqueta POS a cada token, dependendo de seu uso na frase. As tags POS são úteis para atribuir uma categoria sintática, como substantivo ou verbo, a cada palavra. Em spaCy, as tags POS estão disponíveis como um atributo no objeto Token:

```python
# Part of Speech Tagging
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
punctuations = string.punctuation
stopwords = STOP_WORDS
# POS tagging
for i in nlp(review):
    print(i, Fore.GREEN + "=>",i.pos_)
"""
pineapple => NOUN | rind => NOUN | , => PUNCT
lemon => NOUN  | pith => NOUN  | and => CCONJ
orange => NOUN  | blossom => NOUN | start => VERB
off => ADP | the => DET | aroma => NOUN
. => PUNCT  | the => DET | palate => NOUN
be => AUX | a => DET | bit => NOUN
more => ADV | opulent => ADJ | , => PUNCT
....

"""
```



### Text Pre-Processing

#### Stop Words

Stop words são palavras que não tem sentido semântico. Normalmente são removidas ois não tem valor no texto.

Algumas palavras comuns que estão presentes no texto, mas não contribuem para o significado de uma frase. Essas palavras não são importantes para o propósito de recuperação de informações ou processamento de linguagem natural. As palavras irrelevantes mais comuns são ‘the’ e ‘a’.



````python
from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
words = ['I', 'am', 'a', 'writer']
new_words = [word for word in words if word not in english_stops]
# Output: ['I', 'writer']
````

### Corpus

Corpus é o termo técnico para um conjunto de documentos que serão utilizados em um processo de NLP ou Text Mining. É Conjunto de documentos utilizados para processamento de linguagem natural:

### Unigrams. Bigramas e Trigramas

Como achar com Wor2Vec

https://medium.com/@manjunathhiremath.mh/identifying-bigrams-trigrams-and-four-grams-using-word2vec-dea346130eb

Bigramas: Bigram são 2 palavras consecutivas em uma frase. Os bigramas para fase abaixo:

E.g. “The boy is playing football”. 

````
The boy
Boy is
Is playing
Playing football
````

Trigramas: Trigramas são 3 palavras consecutivas em uma frase. Para o exemplo acima, os trigramas serão:

````
The boy is
Boy is playing
Is playing football
````

**Como fazer com Word2Vec**

https://medium.com/@manjunathhiremath.mh/identifying-bigrams-trigrams-and-four-grams-using-word2vec-dea346130eb

## Vetorização

Um **corpus de documentos** pode, portanto, ser representado por uma matriz com **uma linha por documento** e **uma coluna por token** (por exemplo, palavra) ocorrendo no corpus.

Chamamos de **vetorização** o processo geral de transformar uma coleção de documentos de texto em vetores de recursos numéricos. Essa estratégia específica (tokenização, contagem e normalização) é chamada de representação **Saco de palavras** ou "Saco de n-gramas". Os documentos são descritos por ocorrências de palavras, ignorando completamente as informações de posição relativa das palavras no documento.





### TF-IDF

https://www.geeksforgeeks.org/tf-idf-for-bigrams-trigrams/

TF-IDF em NLP significa Frequência do termo - frequência inversa do documento. É um tópico muito popular em Processamento de Linguagem Natural, que geralmente lida com linguagens humanas. 

Durante qualquer processamento de texto, a limpeza do texto (pré-processamento) é vital. Além disso, os dados limpos precisam ser convertidos em um formato numérico onde cada palavra é representada por uma matriz (vetores de palavras). Isso também é conhecido como word embedding (incorporação de palavras).

TF-IDF é feito apartir da multiplicação de TF por IDF

+ Term Frequency (TF) = (Frequency of a term in the document)/(Total number of terms in documents)

+ Inverse Document Frequency (IDF) = log( (total number of documents)/(number of documents with term t))
  + O IDF atua como um fator de equilíbrio e diminui o peso dos termos que ocorrem com muita frequência no conjunto de documentos e aumenta o peso dos termos que ocorrem raramente.

**Exemplo sklearn**

````python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = "wonderful little production filming technique unassuming old time bbc fashion give comforting sometimes discomforting sense realism entire piece actor extremely well chosen michael sheen got polari voice pat truly see seamless editing guided reference williams diary entry well worth watching terrificly written performed piece masterful production one great master comedy life realism really come home little thing fantasy guard rather use traditional dream technique remains solid disappears play knowledge sens particularly scene concerning orton halliwell set particularly flat halliwell mural decorating every surface terribly well done"

tfidf_vec = TfidfVectorizer()
review_tfidf_vec = tfidf_vec.fit_transform(corpus)

review_tfidf_vec.toarray()
"""
array([[0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.19425717, 0.09712859, 0.09712859,
        0.09712859, 0.19425717, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.19425717,
        0.09712859, 0.09712859, 0.19425717, 0.09712859, 0.09712859,
        0.19425717, 0.09712859, 0.19425717, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.19425717, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.09712859, 0.09712859, 0.09712859,
        0.09712859, 0.09712859, 0.29138576, 0.09712859, 0.09712859,
        0.09712859, 0.09712859]])
"""
````

## CountVectorizer

Técnica de Vetorização que conta a coocorrência para cada palavra.



Como a maioria dos documentos geralmente usa um subconjunto muito pequeno de palavras usadas no corpus, a matriz resultante terá muitos valores de recursos que são zeros (normalmente mais de 99% deles).

Por exemplo, uma coleção de 10.000 documentos de texto curto (como e-mails) usará um vocabulário com um tamanho da ordem de 100.000 palavras únicas no total, enquanto cada documento usará de 100 a 1000 palavras únicas individualmente.

Para ser capaz de armazenar tal matriz na memória, mas também para acelerar as operações, as implementações normalmente usam uma representação esparsa, como as implementações disponíveis no pacote `scipy.sparse`.



Podemos imaginar isso como uma matriz bidimensional. Onde a dimensão 1 é todo o vocabulário (1 linha por palavra) e a outra dimensão são os documentos reais, neste caso uma coluna por mensagem de texto.

For example:

|                  | Message 1 | Message 2 | ...  | Message N |
| :--------------- | :-------- | :-------- | :--- | :-------- |
| **Word 1 Count** | 0         | 1         | ...  | 0         |
| **Word 2 Count** | 0         | 0         | ...  | 0         |
| **...**          | 1         | 2         | ...  | 0         |
| **Word N Count** | 0         | 1         | ...  | 1         |

Como há tantas mensagens, podemos esperar muitas contagens zero para a presença dessa palavra naquele documento. Por causa disso, o SciKit Learn produzirá uma [Sparse Matrix] (https://en.wikipedia.org/wiki/Sparse_matrix).

````python
from sklearn.feature_extraction.text import CountVectorizer

corpus = "wonderful little production filming technique unassuming old time bbc fashion give comforting sometimes discomforting sense realism entire piece actor extremely well chosen michael sheen got polari voice pat truly see seamless editing guided reference williams diary entry well worth watching terrificly written performed piece masterful production one great master comedy life realism really come home little thing fantasy guard rather use traditional dream technique remains solid disappears play knowledge sens particularly scene concerning orton halliwell set particularly flat halliwell mural decorating every surface terribly well done"

count_vec = CountVectorizer()# Se colocar 'binary=True' vai se tornar binário, vai ser 0 ou 1
review_count_vec = count_vec.fit_transform(corpus)

review_count_vec.toarray()
"""
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1,
        1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1]])
"""
# 2 and 3 meaning that this word appers 2 or 3 times, to exlcude this, a turn a simply boolean (if a word apear or not) use `CountVectorizer(binary=True)       
````

**Tunar CountVector**

```python
Thus far, we have been using the default parameters of CountVectorizer:

# show default parameters for CountVectorizer
print(vect)
"""
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, vocabulary=None)
"""
# However, the vectorizer is worth tuning, just like a model is worth tuning! Here are a few parameters that you might want to tune:
"""
stop_words: string {'english'}, list, or None (default)
+ If 'english', a built-in stop word list for English is used.
+ If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
+ If None, no stop words will be used.
"""

# remove English stop words
vect = CountVectorizer(stop_words='english')
"""
ngram_range: tuple (min_n, max_n), default=(1, 1)
+ The lower and upper boundary of the range of n-values for different n-grams to be extracted.
+ All values of n such that min_n <= n <= max_n will be used.
"""

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
# max_df: float in range [0.0, 1.0] or int, default=1.0
"""
When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
If float, the parameter represents a proportion of documents.
If integer, the parameter represents an absolute count.
"""

# ignore terms that appear in more than 50% of the documents
vect = CountVectorizer(max_df=0.5)
# min_df: float in range [0.0, 1.0] or int, default=1
"""
When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold. (This value is also called "cut-off" in the literature.)
If float, the parameter represents a proportion of documents.
If integer, the parameter represents an absolute count.
"""

# only keep terms that appear in at least 2 documents
vect = CountVectorizer(min_df=2)
Guidelines for tuning CountVectorizer:

# Use your knowledge of the problem and the text, and your understanding of the tuning parameters, to help you decide what parameters to tune and how to tune them. Experiment, and let the data tell you the best approach!
```



## Text Cleaning

1. Normally any NLP task involves following text cleaning techniques -

````python
from bs4 import BeautifulSoup

soup = BeautifulSoup(review, "html.parser")
review = soup.get_text()
review
````

2. Removal of HTML contents like "< br>".

````python
import re

review = re.sub('\[[^]]*\]', ' ', review)
review = re.sub('[^a-zA-Z]', ' ', review)
review = review.lower()
review

````
3. Removal of punctutions, special characters like '\'.


4. Removal of stopwords like is, the which do not offer much insight.

````python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Steaming
review = [word for word in review if not word in set(stopwords.words('english'))]
review

# Lemmatizer
lem = WordNetLemmatizer()
review = [lem.lemmatize(word) for word in review]
review
````


5. Stemming/Lemmatization to bring back multiple forms of same word to their common root like 'coming', 'comes' into 'come'.

````python
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
review_s = [ps.stem(word) for word in review]
review_s
````

Outros

Explora mais:

https://www.geeksforgeeks.org/python-efficient-text-data-cleaning/?ref=rp

```python
stripped = re.sub('_', '', stripped) 

# Change any white space to one space 
stripped = re.sub('\s+', ' ', stripped) 

# Remove start and end white spaces 
stripped = stripped.strip() 
if stripped != '': 
return stripped.lower()
```



## Redes Neurais em NLP

## Word Embedding

Dicas para fazer Word Embedding com Deep Learning

https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

Etapas:

+ Tokenizar: converter palavras para números
+ pad_sequences: Deixar todos os arrays num mesmo tamanho
+ Embedding: Criar camada de embedding para converter os valores

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Masking, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Let's understand how it works Word Embedding
# Vamos começar com 4 frases
text = ['subha is a good boy', 
        'swati is a very good girl', 
        'yes...we will go home for sure', 
        'India"s gdp is less than USA.']

# Think this is our text which we need to vectorize.
print(text)
"""
['subha is a good boy',
 'swati is a very good girl',
 'yes...we will go home for sure',
 'India"s gdp is less than USA.']
"""

# Tokenizar
tokenizer = Tokenizer(num_words=20)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
sequences
"""
[[4, 1, 2, 3, 5],
 [6, 1, 2, 7, 3, 8],
 [9, 10, 11, 12, 13, 14, 15],
 [16, 17, 18, 1, 19]]
"""
# So we can see we are able to tokenize the textual data.
# Dict do mapeamento de word <=> number
tokenizer.word_index
"""
{'is': 1,  'a': 2,  'good': 3,  'subha': 4,  'boy': 5,  'swati': 6,
 'very': 7,  'girl': 8,  'yes': 9,  'we': 10,  'will': 11,  'go': 12,
 'home': 13,  'for': 14,  'sure': 15,  'india': 16,  's': 17,  'gdp': 18,
 'less': 19,  'than': 20,  'usa': 21}
"""

# Now we can see that the 4 lines in the text have different length; so we need to standardize their length thru a technique called Padding.
# padding='post' means zeroes will be appended after actual text and vice versa if padding='pre'.
# pad_sequences vai deixar todas as 4 frases com mesmo tamanho
data = pad_sequences(sequences, maxlen=8, padding='post')
data
"""
array([[ 4,  1,  2,  3,  5,  0,  0,  0],
       [ 6,  1,  2,  7,  3,  8,  0,  0],
       [ 9, 10, 11, 12, 13, 14, 15,  0],
       [16, 17, 18,  1, 19,  0,  0,  0]], dtype=int32)
"""

# Criando camada de embedding, que é um vetor com vários números
embedding = Embedding(input_dim=20, output_dim=4, mask_zero=True)
masked_output = embedding(data)
print(masked_output.numpy())
"""
# Isso é para apenas uma frase
[[[-0.0432887   0.03164933 -0.00403056 -0.04804061]
  [-0.01957107 -0.01933433  0.03954187  0.03204051]
  [-0.01322449  0.02540943  0.03557068 -0.01157781]
  [ 0.00745217  0.04525444 -0.04387034  0.03782446]
  [-0.0138635  -0.04703224 -0.04861773  0.03589484]
  [ 0.03547157  0.04444233 -0.03654457 -0.00354902]
  [ 0.03547157  0.04444233 -0.03654457 -0.00354902]
  [ 0.03547157  0.04444233 -0.03654457 -0.00354902]]
"""

print("Shape before Embedding: \n", data.shape)
print("Shape after Embedding: \n", masked_output.shape)
"""
Shape before Embedding: 
 (4, 8)
Shape after Embedding: 
 (4, 8, 4)
"""
```



### Word2Vec

[https://medium.com/@everton.tomalok/word2vec-e-sua-import%C3%A2ncia-na-etapa-de-pr%C3%A9-processamento-d0813acfc8ab](https://medium.com/@everton.tomalok/word2vec-e-sua-importância-na-etapa-de-pré-processamento-d0813acfc8ab)

Capturing semantic meaning

Word2Vec é um método estatístico para aprender eficientemente um Word Embedding independente, a partir de um corpus de texto.

Foi desenvolvido por Tomas Mikolov, et al. no Google em 2013 como uma resposta para tornar o treinamento de embeddings baseados em redes neurais mais eficiente.

Com Word Embedding, conseguimos aprender o mapa de características das palavras, através de seus valores numéricos, que são passados ao modelo de treinamento.

Aqui não foi utilizado o vetor one-hot-encoded, mas sim um mapa de vetores multidimensional, que é uma forma algébrica de representar o texto através de coordenadas, onde cada vetor irá representar uma palavra/sentença; desta forma conseguimos utilizar a matemática a nosso favor, para efetuarmos cálculos, tais como a distância entre cada vetor, e diversas outras operações utilizando as propriedades dos vetores.

![](https://miro.medium.com/max/700/1*b06OA4v3x3yhQiFNNzbzow.png)

Agora, a grande sacada foi a seguinte: ao invés de passar apenas informações ao modelo da frequência de cada palavra no texto, foi fornecido muito mais informações que o modelo poderá explorar para aumentar seu “aprendizado” sobre o conjunto de texto.

Agora, com ou em posse de todas as palavras mapeadas como vetores não lineares de múltiplas dimensões, podemos por exemplo, verificar a distância entre cada vetor.

Na imagem abaixo, visualizamos um exemplo gerado, utilizando Word2Vec.

![](https://miro.medium.com/max/700/1*DZq97iVDfEeixcmQleUvqA.png)

Podemos notar que diversos vetores foram agrupados, o que indica uma certa proximidade linguística entre as palavras. Por exemplo, a palavra `is` está mais próxima das palavras `the` , `word2vec` e `yet` do que das palavras `first` e `more`.

Outro exemplo seria, que a diferença entre os vetores `and` e `first` resultaria em algum vetor do agrupamento entre `one` , `second` ,`another` e `final` .

**Others**

## 1.3. Word2Vec

- **What is an iteration-based model?**

  A model that is able to learn one iteration at a time and eventually be able to encode the probability of a word given its context.

- **What is Word2Vec?**

  A model whose parameters are the word vectors. Train the model on a certain objective. At every iteration, we run our model, evaluate the errors and backpropagate the gradients in the model.

- **What are the initial embeddings of Word2Vec model?**

  The embedding matrix is initialized randomly using a Normal or uniform distribution. Then, the embedding of word *i* in the vocabulary is the row *i* of the embedding matrix.

- **What are the two algorithms used by Word2Vec? Explain how they work.**

  Continuous bag-of-words (CBOW)

  Skip-gram

- **What are the two training methods used?**

  Hierarchical softmax

  Negative sampling

- **What is the advantage of Word2Vec over SVD-based methods?**

  Much faster to compute and capture complex linguistic patterns beyond word similarity

- **What is the limitation of Word2Vec?**

  Fails to make use of global co-occurrence statistics. It only relies on local statistics (words in the neighborhood of word *i*).

  E.g.: The cat sat on the mat. Word2Vec doesn't capture if *the* is a special word in the context of cat or just a stop word.

  

### Glove

##  Global vectors for word representations (GloVe)

- **What is GloVe?**

  GloVe aims to combine the SVD-based approach and the context-based skip-gram model.

- **How to build a co-occurence matrix for GloVe? What can we calculate with such a matrix?**

  Let X be a word-word co-occurence matrix (coefficients are the number of times word *i* appears in the context of word *j*). With this matrix, we can compute the probability of word *i* appearing in the context of word j: *Pij = Xij / Xi*

- **How is GloVe built?**

  After building the co-occurence matrix, GloVe computes the ratios of co-occurrence probabilities (non-zero). The intuition is that the word meanings are capture by the ratios of co-occurrence probabilities rather than the probabilities themselves. The global vector models the relationship between two words regarding to the third context word as:

  

  F is designed to be a function of the linear difference between two words *wi* and *wj*. It is an exponential function.

- **What are the pros of GloVe?**

  The GloVe model efficiently leverages global statistical information by training only on non-zero elements in a word-word co-occurence matrix, and produces a vector space with meaningful substructure.

- **What is window classification and why is it important?**

  Natural languages tend to use the same word for very different meanings and we typically need to know the context of the word usage to discriminate between meanings.

  E.g.: 'to sanction' means depending on the context 'to permit' or 'to punish'

  A sequence is a central word vector preceded and succeeded by context word vectors. The number of words in the context is also known as the context window size and varies depending on the problem being solved.

- **How do window size relate to performance?**

  Generally, narrower window size lead to better performance in syntactic tests while wider windows lead to better performance in semantic tests.

### Embedding

Embeddings allow you to turn a categorical variable (e.g., words in a book) into a fixed size vector of real numbers. The key features of embeddings are that they:

Map high dimensional data into a lower-dimensional space

Can be trained to discover relationships between the data points (i.e., the vectors).

By transforming a categorical feature into a numeric value through embeddings, it can be used by your model either as a feature or target value.

Use embeddings to encode categorical features with a large number of categories (e.g., words or sentences) and/or when it’s important to understand how different categories of your categorical feature relate to each other.

### LSTM unit

A long short term memory unit is a special kind of recurrent neural network building block that has a build in ability to 'remember' or 'forget' parts of sequential data. This ability allows a RNN using LSTM units to learn very long range connections in sequential data, by keeping relevant information 'stored' in the unit.

Use in recurrent neural networks.

### Natural language processing (NLP)
Natural language processing (NLP) techniques aim to automatically process, analyze and manipulate (large amounts) of language data like speech and text.



