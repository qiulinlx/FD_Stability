import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

'''
Selecting Environmental subset of data that is California points only'
'''

df= pd.read_csv('Composite.csv')

# There exists a better way to do this 
ca_df = df[
    (df['lat'] >= 32.5) & (df['lat'] <= 42) &
    (df['lon'] >= -124.5) & (df['lon'] <= -114)
]

ca_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

print(ca_df.head(10))

ca_df.to_csv('Ca_dataset.csv', index=False)