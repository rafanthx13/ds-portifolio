
## Esconde index

#Using hide_index() from the style function
planets.head(10).style.hide_index()

## Esconder colunas desnecessárias

planets.head(10).hide_columns(['method','year'])

## Destacar valores masimo ou miniomos da coluna

#Highlight the maximum number for each column
planets.head(10).style.highlight_max(color = 'yellow')
planets.head(10).style.highlight_min(color = 'lightblue')
planets.head(10).style.highlight_null(null_color = 'red')

## Ter um BarChart nas células do dataframe

#Sort the values by the year column then creating a bar chart as the background
planets.head(10).sort_values(by = 'year').style.bar(color= 'lightblue')

# A barra é de acordo com a distribuiçâo da coluna

