# DS PLOTS

<!--

+ /home/rhavel/Documentos/Personal Projects/ds-portifolio/ds-my-snippets/ds-plots/imgs/

https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/

-->


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

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/eda_generate_categorical_feat.png)

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

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/eda_g_categorical_df.png)

## Big Plot to many categorical values

```python
fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x="year", data=df)
plt.title("Number of records per year", fontsize=24)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
```

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/big-plot-categorical.png)

## Lib missing

```python
sns.heatmap(df.isnull(), cbar=False)
```

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/sns-heartmap.png)

```python
import missingno as msno
msno.bar(df)
```

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/missingno.png)

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

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/geoplot-brazil-states.png)

## `eda_bokeh_horiz_bar_ranked`

```python
from bokeh.palettes import Turbo256 
from bokeh.models import ColumnDataSource
from bokeh.palettes import inferno

def eda_bokeh_horiz_bar_ranked(df, column_target, title = '', int_top = 3, second_target = 'state'):
    """
    Generate Bokeh Plot ranking top fists and last value:
        df: data_frame
        column_targe: a column of df inputed
        title: title of plot
        int_top: number of the tops
        column: column of df_in to be placed values in geoplot
        second_targe = 'state'
    """
    ranked = df.sort_values(by=column_target).reset_index(drop = True)
    top_int = int_top
    top = ranked[:top_int].append(ranked[-top_int:]).drop(['geometry'], axis = 1)
    top.index = top.index + 1
    source = ColumnDataSource(data=top)
    list_second_target = source.data[second_target].tolist()
    index_label = list_second_target[::-1] # reverse order label

    p = figure(plot_width=500, plot_height=300, y_range=index_label, 
                toolbar_location=None, title=title)   

    
    # turbo_pallete = Turbo256[0:256:int(256/len(list_second_target) - 2)][::-1] # proportional of number of bars
    p.hbar(y=second_target, right=column_target, source=source, height=0.85, line_color="#000000",
          fill_color=factor_cmap(second_target, palette=inferno(16)[::-1], factors=list_second_target))
    p.x_range.start = 0  # start value of the x-axis

    p.xaxis.axis_label = "value of '" + column_target + "'"

    hover = HoverTool()  # initiate hover tool
    hover.tooltips = [("Value","R$ @{" + column_target + "}" ),   
                       ("Ranking","@index°")]

    hover.mode = 'hline' # set the mode of the hover tool
    p.add_tools(hover)   # add the hover tooltip to the plot

    return p # show in notebook

# Example
# show(eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row['ETANOL HIDRATADO'], column_target = 'avg_price', title = 'AVG Gasolina',
#     int_top = 5, second_target = 'state') )
```

![](https://github.com/rafanthx13/ds-portifolio/blob/master/ds-my-snippets/ds-plots/imgs/eda_bokeh_horiz_bar_ranked.png)
