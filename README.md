# Video-Game-Sales-Analysis
Analysis/visualization of video games sales data and comparison of regression techniques

This repo demonstrates methods of visualizing data and doing regression to extrapolate from known data trends. It uses the Video Game Sales data, which can be found [here](https://www.kaggle.com/gregorut/videogamesales/kernels). The data contains approximately 100,000 entries of video game sales, with features including: Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.

This loads and preprocesses the data.

```Python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("vgsales.csv")

# get an idea of the total number of occurences for important features
# other visualizations

print("Number of games: ", len(df))
publishers = df['Publisher'].unique()
print("Number of publishers: ", len(publishers))
platforms = df['Platform'].unique()
print("Number of platforms: ", len(platforms))
genres = df['Genre'].unique()
print("Number of genres: ", len(genres))

# check to see if there are any null values
print(df.isnull().sum())

# drop them if there are any
df = df.dropna()

# we can get the number of total occurances of a year, and group them, by doing this:
y = df.groupby(['Year']).sum()

# now let's select the sale figures for that year and cast it as an integer
y = y['Global_Sales']
x = y.index.astype(int)
```

This visualizes data to explore possible relationships.

```Python
# now let's try plotting the data
plt.figure(figsize=(12,8))

# this will be a barplot
# call the barplot function from seaborn, provide values
ax = sns.barplot(y=y, x=x)
# set labels and show
# customize these plots a bit
ax.set_xlabel(xlabel='$ Millions', fontsize=16)
ax.set_xticklabels(labels = x, fontsize=12, rotation=50)
ax.set_ylabel(ylabel='Year', fontsize=16)
ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

# if we wanted the counts instead, we could just count. Count returns number of instances,
# not the sums of the values like above
x = df.groupby(['Year']).count()
x = x['Global_Sales']
y = x.index.astype(int)

plt.figure(figsize=(12,8))
colors = sns.color_palette("muted")
ax = sns.barplot(y = y, x = x, orient='h', palette=colors)
ax.set_xlabel(xlabel='Number of releases', fontsize=16)
ax.set_ylabel(ylabel='Year', fontsize=16)
ax.set_title(label='Game Releases Per Year', fontsize=20)

plt.show()

vg_data = pd.read_csv('vgsales.csv')
vg_dummies = pd.get_dummies(vg_data)

print(vg_data.info())
print(vg_data.describe())

# let's choose a cutoff and drop any publishers that have published less than X games
for i in vg_data['Publisher'].unique():
    if vg_data['Publisher'][vg_data['Publisher'] == i].count() < 50:
        vg_data['Publisher'][vg_data['Publisher'] == i] = 'Other'

for i in vg_data['Platform'].unique():
    if vg_data['Platform'][vg_data['Platform'] == i].count() < 90:
        vg_data['Platform'][vg_data['Platform'] == i] = 'Other'

# try plotting the new publisher and platform data
sns.countplot(x='Publisher', data=vg_data)
plt.title("# Games Published By Publisher")
plt.xticks(rotation=-90)
plt.show()

plat_data = vg_data['Platform'].value_counts(sort=False)
sns.countplot(y='Platform', data=vg_data)
plt.title("# Games Published Per Console")
plt.xticks(rotation=-90)
plt.show()

sns.barplot(x='Genre', y='Global_Sales', data=vg_data)
plt.title("Total Sales Per Genre")
plt.show()

# try visualizing the number of games in a specific genre
for i in vg_data['Platform'].unique():
    vg_data['Genre'][vg_data['Platform'] == i].value_counts().plot(kind='line', label=i, figsize=(20, 10), grid=True)

# set the legend and ticks
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=20, borderaxespad=0.)
plt.xticks(np.arange(12), tuple(vg_data['Genre'].unique()))
plt.tight_layout()
plt.show()

```

This chunk  prepares the data from the regressors and then carries out regression using multiple algorithms

```Python
# going to attempt to carry out linear regression and predict the global sales of games
# based off of the sales in North America
X = vg_data.iloc[:, 6].values
y = vg_data.iloc[:, 10].values

# train test split and split the dataframe
X, X2, y, y2 = train_test_split(X, y, test_size=0.1, random_state=5)

# reshape the data into long 2D arrays with 1 column and as many rows as necessary
X_train = X.reshape(-1, 1)
X_test = X2.reshape(-1, 1)
y_train = y.reshape(-1, 1)
y_test = y2.reshape(-1, 1)

# Fit and plot different regressors

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

def plot_regression(classifier):

    plt.scatter(X_train, y_train,color='red')
    plt.plot(X_train, classifier.predict(X_train), color='blue')
    plt.title('(Training set)')
    plt.xlabel('North America Sales')
    plt.ylabel('Global Sales')
    plt.show()

    plt.scatter(X_test, y_test,color='red')
    plt.plot(X_train, classifier.predict(X_train), color='blue')
    plt.title('(Testing set)')
    plt.xlabel('North America Sales')
    plt.ylabel('Global Sales')
    plt.show()

plot_regression(lin_reg)
print("Training set score: {:.2f}".format(lin_reg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lin_reg.score(X_test, y_test)))

DTree_regressor = DecisionTreeRegressor(random_state=5)
DTree_regressor.fit(X_train, y_train)
plot_regression(DTree_regressor)
print("Training set score: {:.2f}".format(DTree_regressor.score(X_train, y_train)))
print("Test set score: {:.2f}".format(DTree_regressor.score(X_test, y_test)))

RF_regressor = RandomForestRegressor(n_estimators=300, random_state=5)
RF_regressor.fit(X_train, y_train)
plot_regression(RF_regressor)
print("Training set score: {:.2f}".format(RF_regressor.score(X_train, y_train)))
print("Test set score: {:.2f}".format(RF_regressor.score(X_test, y_test)))

components = [
    ('scaling', StandardScaler()),
    ('PCA', PCA()),
    ('regression', LinearRegression())
]

pca = Pipeline(components)
pca.fit(X_train, y_train)
plot_regression(pca)

elastic = ElasticNet()
elastic.fit(X_train, y_train)
plot_regression(elastic)
print("Training set score: {:.2f}".format(elastic.score(X_train, y_train)))
print("Test set score: {:.2f}".format(elastic.score(X_test, y_test)))

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
plot_regression(ridge_reg)
print("Training set score: {:.2f}".format(ridge_reg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge_reg.score(X_test, y_test)))

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
plot_regression(lasso_reg)
print("Training set score: {:.2f}".format(lasso_reg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso_reg.score(X_test, y_test)))

ada_reg = AdaBoostRegressor()
ada_reg.fit(X_train, y_train)
plot_regression(ada_reg)
print("Training set score: {:.2f}".format(ada_reg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ada_reg.score(X_test, y_test)))
```



