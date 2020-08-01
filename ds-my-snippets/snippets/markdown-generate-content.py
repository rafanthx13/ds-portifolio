# Snipptes

## Generate Markdown features Content Section

````python
# Example
feats = ["alfa", "beta", "gama"]

def eda_g_md_features(list_features):
  output = ""
  for el in list_features:
    output += "+ `" + el + "`:\n  -  \n"
  print(output)

eda_g_md_features(feats)
````

## Example of Table Of Contents

````markdown
+ [Random-Title](#index01)
````

Where

````markdown
## Random-Title <a id ='index01'></a>
````
