# NLP Snippets

## Mostrar a difenreça do texto antes e depois de text_cleaning

````python
import random 

def compare_text_cleaning(mydf, column1, column2, rows=10):
    """Compare Text after text cleaning
    """
    max_values = len(mydf)
    for i in range(rows):
        anumber = random.randint(0, max_values)
        print('Before:', mydf[column1][anumber])
        print('After :',  mydf[column2][anumber], '\n')
# Example
# compare_text_cleaning(df, 'comment', 'clean_comment')
````
