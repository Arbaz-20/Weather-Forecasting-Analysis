import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        return df
    except pd.errors.EmptyDataError:
        return None
    except pd.errors.ParserError:
        return "ParserError"
    except Exception as e:
        return str(e)

def melt_data(df):
    df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:])
    df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str)
    df1.loc[:, 'Date'] = df1['Date'].apply(lambda x: datetime.strptime(x, '%b %Y'))
    df1.columns = ['Year', 'Month', 'Temperature', 'Date']
    df1.sort_values(by='Date', inplace=True)
    return df1

def plot_temperature_timeline(df1):
    fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, df1['Temperature'].max() + 1])))
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temperature']))
    fig.update_layout(title='Temperature Through Timeline:',
                      xaxis_title='Time', yaxis_title='Temperature in Degrees')
    fig.update_layout(xaxis=go.layout.XAxis(
        rangeselector=dict(
            buttons=list([dict(label="Whole View", step="all"),
                          dict(count=1, label="One Year View", step="year", stepmode="todate")
                          ])),
        rangeslider=dict(visible=True), type="date")
    )
    return fig

def plot_monthly_temperature_box(df1):
    fig = px.box(df1, 'Month', 'Temperature')
    fig.update_layout(title='Warmest, Coldest and Median Monthly Temperature.')
    return fig

def plot_kmeans_clusters(df1):
    sse = []
    target = df1['Temperature'].to_numpy().reshape(-1, 1)
    num_clusters = list(range(1, 10))

    for k in num_clusters:
        km = KMeans(n_clusters=k)
        km.fit(target)
        sse.append(km.inertia_)

    fig = go.Figure(data=[
        go.Scatter(x=num_clusters, y=sse, mode='lines'),
        go.Scatter(x=num_clusters, y=sse, mode='markers')
    ])
    fig.update_layout(title="Evaluation on number of clusters:",
                      xaxis_title="Number of Clusters:",
                      yaxis_title="Sum of Squared Distance",
                      showlegend=False)
    return fig

def fit_kmeans_and_plot(df1):
    km = KMeans(3)
    km.fit(df1['Temperature'].to_numpy().reshape(-1, 1))
    df1.loc[:, 'Temp Labels'] = km.labels_
    fig = px.scatter(df1, 'Date', 'Temperature', color='Temp Labels')
    fig.update_layout(title="Temperature clusters.",
                      xaxis_title="Date", yaxis_title="Temperature")
    return fig

def plot_temperature_histogram(df1):
    fig = px.histogram(x=df1['Temperature'], nbins=200, histnorm='density')
    fig.update_layout(title='Frequency chart of temperature readings:',
                      xaxis_title='Temperature', yaxis_title='Count')
    return fig

def plot_yearly_mean_temperature(df):
    df['Yearly Mean'] = df.iloc[:, 1:].mean(axis=1)
    fig = go.Figure(data=[
        go.Scatter(name='Yearly Temperatures', x=df['YEAR'], y=df['Yearly Mean'], mode='lines'),
        go.Scatter(name='Yearly Temperatures', x=df['YEAR'], y=df['Yearly Mean'], mode='markers')
    ])
    fig.update_layout(title='Yearly Mean Temperature:',
                      xaxis_title='Time', yaxis_title='Temperature in Degrees')
    return fig

def plot_monthly_temperature_history(df1):
    fig = px.line(df1, 'Year', 'Temperature', facet_col='Month', facet_col_wrap=4)
    fig.update_layout(title='Monthly temperature through history:')
    return fig

def plot_seasonal_mean_temperatures(df):
    df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
    df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
    df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
    df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
    seasonal_df = df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']]
    seasonal_df = pd.melt(seasonal_df, id_vars='YEAR', value_vars=seasonal_df.columns[1:])
    seasonal_df.columns = ['Year', 'Season', 'Temperature']

    fig = px.scatter(seasonal_df, 'Year', 'Temperature', facet_col='Season', facet_col_wrap=2, trendline='ols')
    fig.update_layout(title='Seasonal mean temperatures through years:')
    return fig

def predict_temperatures(df1):
    df2 = df1[['Year', 'Month', 'Temperature']].copy()
    df2 = pd.get_dummies(df2)
    y = df2[['Temperature']]
    x = df2.drop(columns='Temperature')

    dtr = DecisionTreeRegressor()
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
    dtr.fit(train_x, train_y)
    pred = dtr.predict(test_x)
    r2 = r2_score(test_y, pred)

    next_year = df1[df1['Year'] == 2017][['Year', 'Month']]
    next_year.Year.replace(2017, 2018, inplace=True)
    next_year = pd.get_dummies(next_year)
    temp_2018 = dtr.predict(next_year)

    temp_2018 = {'Month': df1['Month'].unique(), 'Temperature': temp_2018}
    temp_2018 = pd.DataFrame(temp_2018)
    temp_2018['Year'] = 2018

    forecasted_temp = pd.concat([df1, temp_2018], sort=False).groupby(by='Year')['Temperature'].mean().reset_index()
    fig = go.Figure(data=[
        go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'], mode='lines'),
        go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'], mode='markers')
    ])
    fig.update_layout(title='Forecasted Temperature:',
                      xaxis_title='Time', yaxis_title='Temperature in Degrees')
    return fig, r2
