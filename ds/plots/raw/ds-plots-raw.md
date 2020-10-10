# DS PLOTS

# GENERAL

## Config sns or plot

```python
sns.set() # Com grid m e com fundo umpouco colorido
sns.set(style="whitegrid") # Com grid mas fundo branco

Isso vai mudar tambem os plot do matplot lib

color = sns.color_palette()
sns.set_style('darkgrid')
```

## time_spent

```python
import time

def time_spent(time0):
    t = time.time() - time0
    t_int, t_min = int(t) // 60, t % 60
    return '{} min {:6.3f} s'.format(t_int, t_min) if t_int != 0 else '{:.3f} s'.format(t_min)
```



## Describe with value_counts

```python
def df_value_counts(df0):
    columns = df.columns.tolist()
    for c in columns:
        dtype = df0[c].dtype
        print('Column: "{}" | Type: {}\n==> '.format(c, dtype), end='')
        print('NUMBER OF UNIQUE VALUES: {:,d}\n==> '.format(len(df0[c].unique())), end='')
        # String, Categorical Feature
        if(dtype in ['O', 'object']):
            df_cat = pd.concat([df0[c].value_counts(), df0[c].value_counts(normalize=True)], axis=1).reset_index()
            df_cat.columns = ['values', 'count', 'percentage']
            if(len(df_cat) < 50):
                print('CATEGORICAL VALUES: ', end='')
                for _, row in df_cat.iterrows():
                    print('{} ({:,d} = {:.2%}) ; '.format(row['values'], row['count'], row['percentage']), end='')
            else:
                acount = 0
                print('SOME CAT VALUES: ', end='')
                for _, row in df_cat.iterrows():
                    print('{} ({:,d} = {:.2%}) ; '.format(row['values'], row['count'], row['percentage']), end='')
                    acount += 1
                    if(acount > 15):
                        break
        # Number Feature
        elif(dtype in ['int64','float64']):
            df_int = pd.concat([df[c].value_counts(), df[c].value_counts(normalize=True)], axis=1).reset_index()
            df_int.columns = ['values', 'count', 'percentage']
            text = 'BINARY CAT VALUES: ' if len(df0[c]) == 2 else 'NUMERIC VALUES: '            
            print(text, end='')
            if(len(df_int) < 25):
                # Number but Binary Cat Feature
                for _, row in df_int.iterrows():
                    print('{} ({:,d} = {:.2%}) ; '.format(row['values'], int(row['count']), row['percentage']), end='')
            else:
                # Numeric Feature
                acount = 0
                for _, row in df_int.iterrows():
                    print('{} ({:,d} = {:.2%}) ; '.format(row['values'], int(row['count']), row['percentage']), end='')
                    acount += 1
                    if(acount > 10):
                        break
                # Statistic
                print('\n==> STATISTICS:\n     ==> | ', end='')
                describ = df0[c].describe()
                acount = 0
                for index, value in abc.iteritems():
                    print('{}: {:,.3f} | '.format(index, value), end='')
                    acount += 1
                    if(acount == 4):
                        print('\n     ==> | '.format(index, value), end='')
        print('\n')
# df_value_counts(df)
```

## Describe Transposte

```python
def generate_columns_from_index(topnum):
    adict = {}
    for i in range(topnum):
        adict[i] = 'top' + str(i+1) + '°'
    return adict

def eda_categ_feat_T_rankend(series, top_num):
    return eda_categ_feat_desc_df(series).head(top_num).T.rename(generate_columns_from_index(top_num),axis='columns')
```



# Categorical Feat

## Cat feat top slice

```python
def eda_cat_top_slice_count(s, start=1, end=None, rotate=0):
    # @rotate: 45/80; 
    column, start, threshold = s.name, start - 1, 30
    s = df[column].value_counts()
    lenght = len(s)
    if(end is None):
        end = lenght if lenght <= threshold else threshold
    s = s.reset_index()[start:end]
    s = s.rename(columns = {column: 'count'}).rename(columns = {'index': column,})
    fig, ax = plt.subplots(figsize = (12,4))
    barplot = sns.barplot(x=s[column], y=s['count'], ax=ax)
    # sort by name
    s = s.sort_values(column).reset_index()
    for index, row in s.iterrows():
        barplot.text(row.name, row['count'], '{:,d}'.format(row['count']), color='black', ha="center")
    ax.set_title('Quantity Plot to {}. Top {}°-{}°'.format(column, start+1, end))
    plt.xticks(rotation=rotate)
    plt.show()
# eda_cat_top_slice_count(df['Year'], start=5, end=10, rotate=0)
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_cat_top_slice_count.png)


## EDA categorical feature

````python
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
       @fix_labels: The labes plot in barplot in sorted by values, some times its bugs cuz axis ticks is alphabethic
           if this happens, pass True in fix_labels
       @bar_format: pass {:,.0f} to int
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    if(fix_labels):
        val_concat = val_concat.sort_values(series_name).reset_index()
    
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], '{:,d}'.format(row['quantity']), color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
````

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_generate_categorical_feat.png)

```python
def eda_categ_feat_desc_df(series_categorical):
    """Generate DataFrame with quantity and percentage of categorical series
    @series_categorical = categorical series
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True).apply(lambda x: '{:.2%}'.format(x))
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    return val_concat
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_g_categorical_df.png)



## Horizontal Bar with Seaborn (labeled)

```python
def eda_horiz_plot(df, x, y, title, figsize = (8,5), palette="Blues_d", formating="int"):
    """Using Seaborn, plot horizonal Bar with labels
    !!! Is recomend sort_values(by, ascending) before passing dataframe
    !!! pass few values, not much than 20 is recommended
    """
    f, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x, y=y, data=df, palette=palette)
    ax.set_title(title)
    for p in ax.patches:
        width = p.get_width()
        if(formating == "int"):
            text = int(width)
        else:
            text = '{.2f}'.format(width)
        ax.text(width + 1, p.get_y() + p.get_height() / 2, text, ha = 'left', va = 'center')
    plt.show()

# Example Using:
# top_umber = 10
# df_california = df.query("state == 'CA'").groupby(['city']).count()['date'].sort_values(ascending = False).reset_index().rename({'date': 'count'}, axis = 1)
# list_cities_CA = list(df_california.head(top_umber)['city']) 

# eda_horiz_plot(df_california.head(top_umber), 'count', 'city', 'Rank 10 City death in CA')
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_horiz_plot.png)

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/

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

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/big-plot-categorical.png)





## Tree Map to categorical features

```python
import squarify 
import matplotlib

def tree_map_cat_feat(dfr, column, title='', threshold=1, figsize=(18, 6), alpha=.7):
    """ Print treempa to categorical variables
    Ex: tree_map_cat_feat(df, 'country', 'top countries in country', 200)
    """
    plt.figure(figsize=figsize)
    df_series = dfr[column].value_counts()
    df_mins = df_series[ df_series <= threshold ].sum()
    df_series = df_series[ df_series > threshold ]
    df_series['Others'] = df_mins
    percentages = df_series / df_series.sum()
    alist, mini, maxi = [], min(df_series), max(df_series)
    for i in range(len(df_series)):
        alist.append( df_series.index[i] + '\n{:.2%}'.format(percentages[i]) )
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    cmap = matplotlib.cm.viridis # Blues, plasma, inferno. 
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(i)) for i in df_series]
    squarify.plot(sizes=df_series.values, label=alist, color=colors, alpha=alpha)
    plt.axis('off')
    plt.title(title)
    plt.show()
# tree_map_cat_feat(df, 'top countries', 200)
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/tree_map_cat_feat.png)

# GEO PLOT

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

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/geoplot-brazil-states.png)

## Bokeh Horizontal bar: top and bottom ranking int

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

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_bokeh_horiz_bar_ranked.png)

## Plot eda State with rank (inclusive creation of GeoJSONSource)

```python
### Example
# primary_column = 'SO2'
# target_column = 'total_average_SO2'
# df1 = df.groupby(['District']).mean()[primary_column].reset_index()
# eda_foward_2_plots(df1, primary_column, target_column, "SO total average per district", "The first and last 8 on average for SO")


def eda_foward_2_plots(my_df, primary_column, target_column, first_title, second_title, int_top = 8, location_column = 'District'):
    """
    Execute and show all together:
    !!! primary_columns must to be a float to join to make a GeoSource
    @primary_column: raw column name origined of GroupBy
    @target_column: new name with significance to procees realized by groupby
    @my_df = DataFrame with columns [primary_columns, location_column] with same number of rows of geo_json
    generate_GeoJSONSource_to_districts()
    eda_seoul_districts_geo_plot()
    eda_bokeh_horiz_bar_ranked()
    """
    my_df = my_df.rename({primary_column: target_column}, axis = 1)

    # usado em seoul: necessario quando der problema no .json, entao, vamos colocar os dado direto no json e dele gerar direto o GeoJSONSource
    geo_source = generate_GeoJSONSource_to_districts(my_df, target_column)

    geo = eda_seoul_districts_geo_plot(geo_source, my_df, first_title,
                                       target_column, location_column, palette = inferno(32))

    rank = eda_bokeh_horiz_bar_ranked(my_df, target_column, second_title,
                                      int_top = int_top, second_target = location_column)

    show( row( geo, rank ))
```



# OUTLIERS

## Filter outiliers from a series

Remove outiliers de uma serie, a saida é a serie sem outiliers. útil para fazer análise de feature, pois, se houver muitos outiliers vai atrapalhar os gráficos

```python
def series_remove_outiliers(series):
    # Use IQR Strategy
    # https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
    # def quantils
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    print('Cut Off: below than', lower, 'and above than', upper)
    outliers = series[ (series > upper) | (series < lower)]
    print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
          '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
    # remove outliers
    outliers_removed = [x for x in series if x >= lower and x <= upper]
    print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
    series_no_outiliers = series[ (series <= upper) & (series >= lower) ]
    return series_no_outiliers
```

## Filter Outiliers from a Series of DataFrame

```python
def df_remove_outiliers_from_a_serie(mydf, series_name):
    # Use IQR Strategy
    # https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
    # def quantils
    series = mydf[series_name]
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    print('Cut Off: below than', lower, 'and above than', upper)
    outliers = series[ (series > upper) | (series < lower)]
    print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
          '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
    # remove outliers
    outliers_removed = [x for x in series if x >= lower and x <= upper]
    print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
    mydf = mydf[ (mydf[series_name] <= upper) & (mydf[series_name] >= lower) ]
    return mydf
```

# NUMERICAL FEAT

## describe, boxplot (with label) and distplot to numerical feat

```python
def eda_numerical_feat(series, title="", with_label=True, number_format="", show_describe=False, size_labels=10):
    # Use 'series_remove_outiliers' to filter outiliers
    """ Generate series.describe(), bosplot and displot to a series
    @with_label: show labels in boxplot
    @number_format: 
        integer: 
            '{:d}'.format(42) => '42'
            '{:,d}'.format(12855787591251) => '12,855,787,591,251'
        float:
            '{:.0f}'.format(91.00000) => '91' # no decimal places
            '{:.2f}'.format(42.7668)  => '42.77' # two decimal places and round
            '{:,.4f}'.format(1285591251.78) => '1,285,591,251.7800'
            '{:.2%}'.format(0.09) => '9.00%' # Percentage Format
        string:
            ab = '$ {:,.4f}'.format(651.78) => '$ 651.7800'
    def swap(string, v1, v2):
        return string.replace(v1, "!").replace(v2, v1).replace('!',v2)
    # Using
        swap(ab, ',', '.')
    """
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
    if(show_describe):
        print(series.describe())
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    if(with_label):
        describe = series.describe()
        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
              'Q3': describe.loc['75%']}
        if(number_format != ""):
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
                         size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/eda_numerical_feat.png)

# CLASSIFICATION PLOTS

## Describe Target Y by CatFeat

```python
def describe_y_classify_by_cat_feat(mydf, x, y, title='', classify_content='survivors', labels=['Death', 'Survived']):
    """
    Generate one barplot with quantity and len(x.unique()) pie plots with percentage of x by class of y.unique()
    @classify_content : string that is the meaning of y
    @labels : start from 0, is the meanign of y value
    """
    # Create DataSet
    df1 = df.groupby([x,y]).count().reset_index()
    a_column = df1.columns[2]
    df1 = df1.rename({a_column: "quantity"}, axis=1)
    alist = df1['quantity'].tolist()
    unique_values_x = mydf[x].unique().tolist()
    unique_values_x.sort()
    len_unique_values_y = len(mydf[y].unique().tolist())
    # Create Fig and Axes
    f, ax = plt.subplots(ncols=len(unique_values_x)+1, figsize=(18, 5), sharex=False)
    f.suptitle(title, fontsize=18)
    # BarPlot
    s = sns.barplot(x=x, y='quantity', hue=y, data=df1, ax=ax[0])
    count, by_hue = 0, 0
    for index, row in df1.iterrows():
        axis_x = count - 0.20 if index % 2 == 0 else count + 0.20
        by_hue += 1
        if(by_hue == len_unique_values_y):
            count += 1
            by_hue = 0
            # print(axis_x) ## DEBUG
        s.text(axis_x, row['quantity'], '{:,d}'.format(int(row['quantity'])), color='black', ha="center")
    # Query DF
    hue_count = 0
    for i in range(len(unique_values_x)):
        df1.query('{} == "{}"'.format(x, unique_values_x[i])).plot.pie(y='quantity', figsize=(18, 5), autopct='%1.2f%%',
                                    labels = ['{} = {}'.format(labels[0], str(alist[i+hue_count])),
                                              '{} = {}'.format(labels[1], str(alist[i+hue_count+1]))],
                                    title='{} {} {} (Total = {})'.format(x, unique_values_x[i], classify_content ,str(alist[i] + alist[i+1])),
                                    ax=ax[i+1], labeldistance=None)
        hue_count += 1
    plt.show()
    # return df1 ## DEBUG
    
# describe_y_classify_by_cat_feat(df, x='embarked', y='survived', title='survived by sex')
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/describe_y_classify_by_cat_feat.png)

## Describe Target Y by NumericFeat

```python
def describe_y_classify_numeric_feature(mydf, x, y, title='', with_swarmp=False):
    f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5), sharex=False)
    # Box and Violin Plots
    sns.boxplot(y=x, x=y, data=mydf, ax=ax1)
    sns.violinplot(y=x, x=y, data=mydf, ax=ax2)
    if(with_swarmp):
        sns.swarmplot(x=y, y=x, data=mydf, ax=ax2, palette='rocket')
    # HistogramPlot
    y_unique_values = mydf[y].unique().tolist()
    for u in y_unique_values:
        adf = mydf.query("{} == {}".format(y, u))
        sns.distplot(adf[x], ax=ax3)
    # Set Titles
    if(not title):
        f.suptitle('{} by {}'.format(y,x), fontsize=18)
    else:
        f.suptitle(title, fontsize=18)
    ax1.set_title("BoxPlot")
    ax2.set_title("ViolinPlot")
    ax3.set_title("HistogramPlot")
    plt.show()
    
# describe_y_classify_numeric_feature(df, x='fare', y='survived')
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/describe_y_classify_numeric_feature.png)

## Curva ROC vários Modelos

```python
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/roc-curves-tops.png)

## Matrix de Confusão colorida com report

```python
import itertools

# Create a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # put target_names
    #  print(classification_report(y_test,y_pred, target_names=['Not Purchased', 'Purchased']))
    
 
```

Uso

```python
from sklearn.metrics import confusion_matrix

y_pred = log_reg.predict(X_test)
y_pred

labels = ['No Fraud', 'Fraud']

confusion_mtx = confusion_matrix(y_test, y_pred)

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(confusion_mtx, labels, title="Random UnderSample \n Confusion Matrix")
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/confusion-matrix-colored.png)

Apesar disso, é preferível so o report mesmo que informa o f1, pois com ele conseguimos avaliar melhor classifiacções desbalanceadas.

# CORRELATION

### HeatMap seaborn correlation

```python
f, ax = plt.subplots(figsize=(16,10))

sub_sample_corr = new_df.corr()

mask = np.zeros_like(sub_sample_corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax, mask=mask)
ax.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/heat-map-correlation.png)

### Ranking Correlations seaborn

```python
# Generate Ranking of correlations (boht positives, negatives)

corr = new_df.corr().abs() # Show greater correlations both negative and positive
dict_to_rename = {0: "value", "level_0": "feat1", "level_1": "feat2"} # Rename DataFrame
s = corr.unstack().reset_index().rename(dict_to_rename, axis = 1) # Restructure dataframe

# remove rows thas like 'x' | 'x' 
s_to_drop = s[(s['feat1'] == s['feat2'])].index 
s = s.drop(s_to_drop).reset_index()

s = s.sort_values(by="value", ascending=False).drop("index", axis=1) # sort

# remove rows like "x1 , x2 = y", "x2 , x1 = y". Duplicate cuz is inverted feats
s = s.drop_duplicates('value') 
s[:10]
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/ranking-correlation.png)

### Sorted Corr to target column

```python
def plot_top_rank_correlation(my_df, column_target):
    corr_matrix = my_df.corr()
    top_rank = len(corr_matrix)
    f, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=False)

    ax1.set_title('Top Correlations to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(len(corr_matrix), column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/plot_top_rank_correlation_2.png)

### Top and Bottom Correlation to target_column

To big number of features

```python
def plot_top_bottom_rank_correlation(my_df, column_target, top_rank=5):
    corr_matrix = my_df.corr()
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)

    ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
    ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 8}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
    cols_bot  = cols_bot.insert(0, column_target)
    print(cols_bot)
    cm = np.corrcoef(my_df[cols_bot].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols_bot.values,
                     xticklabels=cols_bot.values, mask=mask, ax=ax2)
    
    plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/plot_top_rank_correlation.png)

# NORMAL DISTRIBUTION

### Test Normality: Skewness and Kutoise

```python
from scipy.stats import norm
from scipy import stats

def test_normal_distribution(serie, thershold=0.4):
    series_name = serie.name
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    f.suptitle('{} is a Normal Distribution?'.format(series_name), fontsize=18)
    ax1.set_title("Histogram to " + series_name)
    ax2.set_title("Q-Q-Plot to "+ series_name)
    
    # calculate normal distrib. to series
    mu, sigma = norm.fit(serie)
    print('Normal dist. (mu= {:,.2f} and sigma= {:,.2f} )'.format(mu, sigma))
    
    # skewness and kurtoise
    skewness = serie.skew()
    kurtoise = serie.kurt()
    print("Skewness: {:,.2f} | Kurtosis: {:,.2f}".format(skewness, kurtoise))
    # evaluate skeness
    # If skewness is less than −1 or greater than +1, the distribution is highly skewed.
    # If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.
    # If skewness is between −½ and +½, the distribution is approximately symmetric.
    pre_text = '\t=> '
    if(skewness < 0):
        text = pre_text + 'negatively skewed or left-skewed'
    else:
        text =  pre_text + 'positively skewed or right-skewed\n'
        text += pre_text + 'in case of positive skewness, log transformations usually works well.\n'
        text += pre_text + 'np.log(), np.log1(), boxcox1p()'
    if(skewness < -1 or skewness > 1):
        print("Evaluate skewness: highly skewed")
        print(text)
    if( (skewness <= -0.5 and skewness > -1) or (skewness >= 0.5 and skewness < 1)):
        print("Evaluate skewness: moderately skewed")
        print(text)
    if(skewness >= -0.5 and skewness <= 0.5):
        print('Evaluate skewness: approximately symmetric')
    # evaluate kurtoise
    #     Mesokurtic (Kurtoise next 3): This distribution has kurtosis statistic similar to that of the normal distribution.
    #         It means that the extreme values of the distribution are similar to that of a normal distribution characteristic. 
    #         This definition is used so that the standard normal distribution has a kurtosis of three.
    #     Leptokurtic (Kurtosis > 3): Distribution is longer, tails are fatter. 
    #         Peak is higher and sharper than Mesokurtic, which means that data are heavy-tailed or profusion of outliers.
    #         Outliers stretch the horizontal axis of the histogram graph, which makes the bulk of the data appear in a 
    #         narrow (“skinny”) vertical range, thereby giving the “skinniness” of a leptokurtic distribution.
    #     Platykurtic: (Kurtosis < 3): Distribution is shorter, tails are thinner than the normal distribution. The peak
    #         is lower and broader than Mesokurtic, which means that data are light-tailed or lack of outliers.
    #         The reason for this is because the extreme values are less than that of the normal distribution.
    print('evaluate kurtoise')
    if(kurtoise > 3 + thershold):
        print(pre_text + 'Leptokurtic: anormal: Peak is higher')
    elif(kurtoise < 3 - thershold):
        print(pre_text + 'Platykurtic: anormal: The peak is lower')
    else:
        print(pre_text + 'Mesokurtic: normal: the peack is normal')
    
    # shapiro-wilki test normality
    # If the P-Value of the Shapiro Wilk Test is larger than 0.05, we assume a normal distribution
    # If the P-Value of the Shapiro Wilk Test is smaller than 0.05, we do not assume a normal distribution
    #     print("Shapiro-Wiki Test: Is Normal Distribution? {}".format(stats.shapiro(serie)[1] < 0.01) )
    #     print(stats.shapiro(serie))

    
    # ax1 = histogram
    sns.distplot(serie , fit=norm, ax=ax1)
    ax1.legend(['Normal dist. ($\mu=$ {:,.2f} and $\sigma=$ {:,.2f} )'.format(mu, sigma)],
            loc='best')
    ax1.set_ylabel('Frequency')
    # ax2 = qq-plot
    stats.probplot(serie, plot=ax2)
    plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/test_normal_distribution.png)



# MISSING DATA

## DF Top missing columns

```python
def df_rating_missing_data(my_df):
    """Create DataFrame with Missing Rate
    """
    # get sum missing rows and filter has mising values
    ms_sum = my_df.isnull().sum()
    ms_sum = ms_sum.drop( ms_sum[ms_sum == 0].index )
    # get percentage missing ratio and filter has mising values
    ms_per = (my_df.isnull().sum() / len(my_df))
    ms_per = ms_per.drop( ms_per[ms_per == 0].index)
    # order by
    ms_per = ms_per.sort_values(ascending=False)
    ms_sum = ms_sum.sort_values(ascending=False)
    # format percentage
    ms_per = ms_per.apply(lambda x: '{:.3%}'.format(x))
    return pd.DataFrame({'Missing Rate' : ms_per, 'Count Missing': ms_sum})  
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/df_rating_missing_data.png)

# REGRESSION

## Plot pointPlot Regression scores

```python
def plot_model_score_regression(models_name_list, model_score_list, title=''):
    fig = plt.figure(figsize=(15, 6))
    ax = sns.pointplot( x = models_name_list, y = model_score_list, 
        markers=['o'], linestyles=['-'])
    for i, score in enumerate(model_score_list):
        ax.text(i, score + 0.002, '{:.4f}'.format(score),
                horizontalalignment='left', size='large', 
                color='black', weight='semibold')
    plt.ylabel('Score', size=20, labelpad=12)
    plt.xlabel('Model', size=20, labelpad=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xticks(rotation=70)
    plt.title(title, size=20)
    plt.show()
    
plot_model_score_regression(list(scores.keys()), [score for score, _ in scores.values()])
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/plot_model_score_regression.png)

### SHow final scores model

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def evaluate_regression(y_pred, y_test, title=''):
    if(title):
        print(title)
    print('MAE  : {:14,.3f}'.format(mean_absolute_error(y_pred, y_test)))
    print('MSE  : {:14,.3f}'.format(mean_squared_error(y_pred, y_test)))
    print('RMSE : {:14,.3f}'.format(np.sqrt(mean_squared_error(y_pred, y_test))))
    print('RMSLE: {:14,.3f}'.format(np.sqrt(mean_squared_log_error(np.absolute(y_pred), y_test))))
    print('R2   : {:14,.3f}'.format(r2_score(y_pred, y_test)))
```



# EDA

## EDA Describe X by Y

```python
def describe_y_numeric_by_x_cat_boxplot(dtf, x_feat, y_target, title='', figsize=(15,5), rotatioon_degree=0):
    """ Generate a quickly boxplot  to describe each Ŷ by each categorical value of x_feat
    """
    the_title = title if title != '' else '{} by {}'.format(y_target, x_feat)
    fig, ax1 = plt.subplots(figsize = figsize)
    sns.boxplot(x=x_feat, y=y_target, data=dtf, ax=ax1)
    ax1.set_title(the_title, fontsize=18)
    plt.xticks(rotation=rotatioon_degree)
    plt.show()
 # Example
 # describe_y_numeric_by_x_cat_boxplot(df, 'score', 'comment_len', figsize=(10,5))
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/describe_y_by_x_cat_boxplot.png)

# NLP

## Words Distribution

```python
def plot_words_distribution(mydf, target_column, title='Words distribution', x_axis='Words in column'):
    # adaptade of https://www.kaggle.com/alexcherniuk/imdb-review-word2vec-bilstm-99-acc
    # def statistics
    len_name = target_column+'_len'
    mydf[len_name] = np.array(list(map(len, mydf[target_column])))
    sw = mydf[len_name]
    median = sw.median()
    mean   = sw.mean()
    mode   = sw.mode()[0]
    # figure
    fig, ax = plt.subplots()
    sns.distplot(df['description_len'], bins=df['description_len'].max(),
                hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,
                kde_kws={"color": "black", 'linewidth': 3})
    ax.set_xlim(left=0, right=np.percentile(df['description_len'], 95)) # Dont get outiliers
    ax.set_xlabel(x_axis)
    ymax = 0.014
    plt.ylim(0, ymax)
    # plot vertical lines for statistics
    ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)
    ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
    ax.plot([median, median], [0, ymax], '--',
            label=f'median = {median:.2f}', linewidth=4)
    ax.set_title(title, fontsize=20)
    plt.legend()
    plt.show()
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/plot_words_distribution.png)

## Top 1,2,3 Words

```python
from sklearn.feature_extraction.text import CountVectorizer

def ngrams_corpus_counter(corpus,ngram_range,n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df

def plot_ngrams_words(series_words, title='Top 10 words'):
    """Plot 3 graphs
    @series_words: a series where each row is a set of words
    """
    # Generate
    df_1_grams = ngrams_corpus_counter(series_words, (1,1), 10)
    df_2_grams = ngrams_corpus_counter(series_words, (2,2), 10)
    df_3_grams = ngrams_corpus_counter(series_words, (3,3), 10)

    fig, axes = plt.subplots(figsize = (18,4), ncols=3)
    fig.suptitle(title)

    sns.barplot(y=df_1_grams['text'], x=df_1_grams['count'],ax=axes[0])
    axes[0].set_title("1 grams")

    sns.barplot(y=df_2_grams['text'], x=df_2_grams['count'],ax=axes[1])
    axes[1].set_title("2 grams",)

    sns.barplot(y=df_3_grams['text'], x=df_3_grams['count'],ax=axes[2])
    axes[2].set_title("3 grams")

    plt.show()
    
# Example: plot_ngrams_words(df['comment'])
```

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/ngrams_plot.png)

tree

## Memory Usage

```python
def reduce_memory_usage(df):
    """ The function will reduce memory of dataframe
    Note: Apply this function after removing missing value"""
    intial_memory = df.memory_usage().sum()/1024**2
    print('Intial memory usage:',intial_memory,'MB')
    for col in df.columns:
        mn = df[col].min()
        mx = df[col].max()
        if df[col].dtype != object:            
            if df[col].dtype == int:
                if mn >=0:
                    if mx < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif mx < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            if df[col].dtype == float:
                df[col] =df[col].astype(np.float32)
    
    red_memory = df.memory_usage().sum()/1024**2
    print('Memory usage after complition: ',red_memory,'MB')

# reduce_memory_usage(df)
```

## RANDOM

### Describe Horizontal

````
def describe_horizontal_serie(serie):
    adec = serie.describe()
    adtype = serie.dtype
    adf = pd.DataFrame(data=adec.values).T
    adf.columns = adec.index
    adf.index = pd.Index([adec.name])
    if(adtype in ['int64']):
        alist = ['min', '25%', '50%', '75%', 'max']
        for c in alist:
            adf[c] = adf[c].astype('int64')
            adf[c] = adf[c].map(lambda x: "{:,d}".format(int(x)))
    adf['count'] = adf['count'].map(lambda x: "{:,d}".format(int(x)))
    return adf
````

![](/home/rhavel/Documentos/Personal Projects/ds-portifolio/ds/plots/imgs/describe_horizontal.png)

