# DS PLOTS



## `eda_generate_categorical_feat`

````python
def eda_categ_feat_desc_plot(series_categorical):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (17,5), ncols=2, nrows=1) # figsize = (width, height)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
````

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/eda_generate_categorical_feat.png)

```python
def eda_categ_feat_desc_df(series_categorical):
    """Generate DataFrame with quantity and percentage of categorical series
    @series_categorical = categorical series
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    return val_concat
```



![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/eda_g_categorical_df.png)

## Lib missing

`sns.heatmap(df.isnull(), cbar=False)`

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/sns-heartmap.png)



![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/missingno.png)

## Plot Brazil States with bokeh

````python
import geopandas as gpd

# get geojson
brazil_geojson = gpd.read_file('../input/brazilstatejsongeogrpah/brazil-states.geojson')

# ordena
brazil_geojson = brazil_geojson.sort_values('name')
brazil_geojson = brazil_geojson.drop(['id', 'regiao_id', 'regiao_id', 'codigo_ibg', 'cartodb_id',
                                      'created_at', 'updated_at'], axis = 1)
brazil_geojson.head()

# JOIN: df <=> brazil_json | df1 devera ter a mesma qtd de rows com o seu 'state' com mesmo valor de 'state_sigla
geoplot = brazil_geojson.merge(df1, left_on = 'state_sigla', right_on = 'state')
geoplot = geoplot.drop(['start_date', 'end_date'], axis = 1)
geoplot.info()


from bokeh.models import GeoJSONDataSource

geosource = GeoJSONDataSource(geojson = geoplot.to_json())
geosource
````

````python
# Antes você junta o geo_json com o df e faz

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, HoverTool, ColorBar
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_brazil_state_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1):
    """
    Generate Bokeh Plot to Brazil States:
        geosource: GeoJSONDataSource of Bokeh
        df_in: DataSet before transformed in GeoJSONDataSource
        title: title of plot
        column: column of df_in to be placed values in geoplot
        state_column: indicate column with names of States
        low = (optional) min value of range of color spectre
        high = (optional) max values of range of color spectre
        palette: (optional) can be magma, viridis, civis, inferno e etc.. (with number os colors)
            Example: cividis(8) (8 colors to classify), cividis(256)  (256, more colors to clasify)
    """
    if high == -1:
        high = max(df_in[column])
    if low == -1:
        low = min(df_in[column])
    if palette == -1:
        palette = inferno(32)
        
    palette = palette[::-1]
    color_mapper = LinearColorMapper(palette = palette, low = low, high = high)
    
    hover = HoverTool(tooltips = [ ('State','@{'+state_column+'}'), (column, '@{'+column+'}{%.2f}')],
                  formatters={'@{'+column+'}' : 'printf'})

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width = 300, height = 20, 
                         border_line_color=None, location = (0,0),  orientation = 'horizontal')

    p = figure(title = title, plot_height = 430, plot_width = 330, tools = [hover])

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

    p.patches('xs','ys', source = geosource, line_color = 'black', line_width = 0.25,
              fill_alpha = 1, fill_color = {'field' : str(column), 'transform' : color_mapper})

    p.add_layout(color_bar, 'below')
    return p
````

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/geoplot-brazil-states.png)

