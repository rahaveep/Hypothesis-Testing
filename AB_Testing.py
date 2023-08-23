import matplotlib.pyplot as plt
import pandas as pd


market_data = pd.read_csv('Marketing Campaign.csv')

#DATA ANALYSIS

print(market_data.shape,market_data.head())

#Total Sales
market_data['SalesInThousands'].describe()

ax = market_data.groupby(
    'Promotion'
).sum()[
    'SalesInThousands'
].plot.pie(
    figsize=(7, 7),
    autopct='%1.0f%%'
)

ax.set_ylabel('')
ax.set_title('sales distribution across different promotions')

plt.show()

#Market Size
market_data.groupby('MarketSize').count()['MarketID']

ax = market_data.groupby([
    'Promotion', 'MarketSize'
]).count()[
    'MarketID'
].unstack(
    'MarketSize'
).plot(
    kind='bar',
    figsize=(12,10),
    grid=True,
)

ax.set_ylabel('count')
ax.set_title('breakdowns of market sizes across different promotions')

plt.show()

ax = market_data.groupby([
    'Promotion', 'MarketSize'
]).sum()[
    'SalesInThousands'
].unstack(
    'MarketSize'
).plot(
    kind='bar',
    figsize=(12,10),
    grid=True,
    stacked=True
)

ax.set_ylabel('Sales (in Thousands)')
ax.set_title('breakdowns of market sizes across different promotions')

plt.show()

#Age of Store
ax = market_data.groupby(
    'AgeOfStore'
).count()[
    'MarketID'
].plot(
    kind='bar',
    color='skyblue',
    figsize=(10,7),
    grid=True
)

ax.set_xlabel('age')
ax.set_ylabel('count')
ax.set_title('overall distributions of age of store')

plt.show()

ax = market_data.groupby(
    ['AgeOfStore', 'Promotion']
).count()[
    'MarketID'
].unstack(
    'Promotion'
).iloc[::-1].plot(
    kind='barh',
    figsize=(12,15),
    grid=True
)

ax.set_ylabel('age')
ax.set_xlabel('count')
ax.set_title('overall distributions of age of store')

plt.show()

market_data.groupby('Promotion').describe()['AgeOfStore']

#Week Number
market_data.groupby('week').count()['MarketID']

market_data.groupby(['Promotion', 'week']).count()['MarketID']

ax1, ax2, ax3 = market_data.groupby(
    ['week', 'Promotion']
).count()[
    'MarketID'
].unstack('Promotion').plot.pie(
    subplots=True,
    figsize=(24, 8),
    autopct='%1.0f%%'
)

ax1.set_ylabel('Promotion #1')
ax2.set_ylabel('Promotion #2')
ax3.set_ylabel('Promotion #3')

ax1.set_xlabel('distribution across different weeks')
ax2.set_xlabel('distribution across different weeks')
ax3.set_xlabel('distribution across different weeks')

plt.show()

#STATISTICAL SIGNIFICANCE
import numpy as np
from scipy import stats

mean = market_data.groupby('Promotion').mean()['SalesInThousands']
print(mean)

std = market_data.groupby('Promotion').std()['SalesInThousands']
print(std)

ns = market_data.groupby('Promotion').count()['SalesInThousands']
print(ns)

#Promotion 1 VS 2
t_1_vs_2 = (
    mean.iloc[0] - mean.iloc[1]
)/ np.sqrt(
    (std.iloc[0]**2/ns.iloc[0]) + (std.iloc[1]**2/ns.iloc[1])
)

d_1_vs_1 = ns.iloc[0] + ns.iloc[1] - 2

p_1_vs_2 = (1 - stats.t.cdf(t_1_vs_2, df=d_1_vs_1))*2

print(t_1_vs_2)
print(p_1_vs_2)

t, p = stats.ttest_ind(
    market_data.loc[market_data['Promotion'] == 1, 'SalesInThousands'].values,
    market_data.loc[market_data['Promotion'] == 2, 'SalesInThousands'].values,
    equal_var=False
)
print(t)
print(p)

#Promotion 1 vs 3
t_1_vs_3 = (
    mean.iloc[0] - mean.iloc[2]
)/ np.sqrt(
    (std.iloc[0]**2/ns.iloc[0]) + (std.iloc[2]**2/ns.iloc[2])
)

d_1_vs_3 = ns.iloc[0] + ns.iloc[1] - 2

p_1_vs_3 = (1 - stats.t.cdf(t_1_vs_3, df=d_1_vs_3))*2
print(t_1_vs_3)
print(p_1_vs_3)

t, p = stats.ttest_ind(
    market_data.loc[market_data['Promotion'] == 1, 'SalesInThousands'].values,
    market_data.loc[market_data['Promotion'] == 3, 'SalesInThousands'].values,
    equal_var=False
)
print(t)
print(p)
