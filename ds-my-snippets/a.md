# Snipptes

## Generate Markdown features Content

````python
feats = ["alfa", "beta", "gama"]

def eda_g_md_features(list_features):
  output = ""
  for el in list_features:
    output += "+ `" + el + "`:\n  -  \n"
  print(output)

eda_g_md_features(feats)
````
