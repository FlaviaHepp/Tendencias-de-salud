
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
template = "plotly_dark"
plt.style.use('dark_background')
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import os
import arrow
from geocoder import arcgis
import warnings
from plotly import express
warnings.filterwarnings('ignore')
#Regresión
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('gapminder.csv')

#df = pd.read_csv(filepath_or_buffer=DATA) #Original,parse_dates=['Tiempo de observación']

print(df.head())

df = df.drop_duplicates(ignore_index=True)

time_start = arrow.now()
df['latlng'] = df['Region'].apply(func=lambda x: arcgis(x).latlng)
df['latitude'] = df['latlng'].apply(func=lambda x: x[0])
df['longitude'] = df['latlng'].apply(func=lambda x: x[1])

print('{} done'.format(arrow.now() - time_start))

warnings.filterwarnings(action='ignore', category=FutureWarning)
for column in ['Country', 'LifeExpectancy', 'FertilityRate', 'Population']:
    express.scatter_mapbox(mapbox_style='open-street-map', data_frame=df, lat='latitude', lon='longitude', zoom=1, hover_name='Region',
                           color=column, template = template).show()
    

print(df)

print(df.describe().T)

print(df.info())

print(df['Population'].unique())

numeric_population = pd.to_numeric(df['Population'], errors='coerce')

nan_population_rows = df[numeric_population.isna()]
print(nan_population_rows)

df['Population'] = df['Population'].str.replace(',', '').astype(int)

print(df['Population'])
print(df['Population'].dtype)

#EDA + Análisis Comparativo
#Análisis univariado

countries_per_region = df.groupby('Region')['Country'].nunique().reset_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(countries_per_region['Region'], countries_per_region['Country'], color='skyblue', edgecolor = "blueviolet")
plt.title('Número de países en cada región\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Región\n')
plt.ylabel('Número de países\n')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 1), va='bottom')

plt.tight_layout()
plt.show()

fig = px.choropleth(df, 
                    locations="Country", 
                    locationmode='country names',
                    color="LifeExpectancy",
                    hover_name="Country",
                    hover_data={"LifeExpectancy": True,
                               "Population": True},
                    title="Mapa mundial de esperanza de vida\n",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    template = template)

fig.show()

fig = px.choropleth(df, 
                    locations="Country", 
                    locationmode='country names',
                    color="Population",
                    hover_name="Country",
                    hover_data={"LifeExpectancy": True,
                               "Population": True},
                    title="Mapa mundial de población\n",
                    color_continuous_scale=px.colors.sequential.Plasma, 
                    template = template)

fig.show()

#Esperanza de vida más alta y más baja por país
country_highest_life_expectancy = df.loc[df['LifeExpectancy'].idxmax()]['Country']
highest_life_expectancy = df['LifeExpectancy'].max()

country_lowest_life_expectancy = df.loc[df['LifeExpectancy'].idxmin()]['Country']
lowest_life_expectancy = df['LifeExpectancy'].min()

print("País con mayor esperanza de vida:", country_highest_life_expectancy, "con una esperanza de vida de", highest_life_expectancy)
print("Países con menor esperanza de vida:", country_lowest_life_expectancy, "con una esperanza de vida de", lowest_life_expectancy)

average_life_expectancy_by_region = df.groupby('Region')['LifeExpectancy'].mean().reset_index()

average_life_expectancy_by_region = average_life_expectancy_by_region.sort_values(by='LifeExpectancy', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(average_life_expectancy_by_region['Region'], average_life_expectancy_by_region['LifeExpectancy'], color='skyblue', edgecolor= "deeppink")
plt.title('Esperanza de vida promedio por región\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Región\n')
plt.ylabel('Esperanza de vida media\n')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

average_fertility_rate_by_region = df.groupby('Region')['FertilityRate'].mean().reset_index()

average_fertility_rate_by_region = average_fertility_rate_by_region.sort_values(by='FertilityRate', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(average_fertility_rate_by_region['Region'], average_fertility_rate_by_region['FertilityRate'], color='skyblue', edgecolor= "mediumblue")
plt.title('Tasa de fertilidad promedio por región\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Región\n')
plt.ylabel('Tasa de fertilidad promedio\n')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Análisis bivariado y multivariado
sns.pairplot(df,corner=True)
plt.show()

#Correlación
correlation_df = df[['LifeExpectancy', 'FertilityRate', 'Population']]

correlation_matrix = correlation_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap= "Blues")
plt.title('Mapa de calor de correlación\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Escala mín. máx.
columns_to_scale = ['Population', 'LifeExpectancy', 'FertilityRate']

scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), columns=columns_to_scale)

print(df_scaled.head())

#Esperanza de vida por población y tasa de fertilidad
X = df_scaled[['Population', 'FertilityRate']]
y = df_scaled['LifeExpectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error medio cuadrado:", mse)
print("R-cuadrado:", r2)
print("\nCoeficientes:")
for i, feature in enumerate(X.columns):
    print(feature, ":", model.coef_[i])
print("Interceptar:", model.intercept_)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='skyblue', edgecolor = "blue")
plt.title('Esperanza de vida real versus prevista (regresión lineal)\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Esperanza de vida real\n')
plt.ylabel('Esperanza de vida prevista\n')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()