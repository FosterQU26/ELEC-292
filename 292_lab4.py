import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#Question 1
dataset = pd.read_csv('heart.csv')
data = dataset.iloc[:, : -1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))
data.hist(ax=ax.flatten()[:13])

fig.tight_layout()
plt.show()
#Question 3
dataset = pd.read_csv('heart.csv')
data = dataset.iloc[:, : -1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))
for i in range(0, 12):
    ax.flatten()[i].hist(data.iloc[:, i])
    ax.flatten()[i].set_title(data.columns[i], fontsize=15)

fig.tight_layout()
plt.show()
#Question 4
dataset = pd.read_csv('heart.csv')
data = dataset.iloc[:, : -1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))

data.plot(ax=ax.flatten()[:13], kind='density', subplots=True, sharex=False)

fig.tight_layout()
plt.show()
#Question 5
dataset = pd.read_csv('heart.csv')
data = dataset.iloc[:, : -1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))

data.plot(ax=ax.flatten()[:13], kind='box', subplots=True, sharex=False, sharey=False)

fig.tight_layout()
plt.show()
#Question 6
dataset = pd.read_csv('heart.csv')
data = dataset.iloc[:, : -1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols= 13, nrows= 13, figsize=(30,30))

pd.plotting.scatter_matrix(data, ax = ax)

fig.tight_layout()
plt.show()

#Question 8


wine_data = pd.read_csv("winequalityN.csv")
wine_data['quality'] = (wine_data['quality'] >= 8).astype(int)
features = wine_data.iloc[:, 1:]
target = wine_data['quality']

scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, init='pca')
tsne_components = tsne.fit_transform(standardized_features)

fig, ax = plt.subplots(figsize=(10, 10))

colors = ['pink', 'red']
labels = ['low-quality', 'quality']

for label_value, color in zip(set(target), colors):
    ax.scatter(
        tsne_components[target == label_value, 0],
        tsne_components[target == label_value, 1],
        c=color,
        s=60,
        label=labels[label_value]
    )

ax.set_xlabel('t-SNE Component - 1', fontsize=15)
ax.set_ylabel('t-SNE Component - 2', fontsize=15)
ax.set_title('t-SNE of Wine Quality Dataset', fontsize=15)
ax.legend(labels, fontsize=15)

plt.show()



wine_data = pd.read_csv("winequalityN.csv")
wine_data['quality'] = (wine_data['quality'] >= 8).astype(int)
features = wine_data.iloc[:, 1:]
target = wine_data['quality']

scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized_features)

fig, ax = plt.subplots(figsize=(10, 10))

colors = ['pink', 'red']
labels = ['low-quality', 'quality']

for label_value, color in zip(set(target), colors):
    ax.scatter(
        principal_components[target == label_value, 0],
        principal_components[target == label_value, 1],
        c=color,
        s=60,
        label=labels[label_value]
    )

ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('PCA of Wine Quality Dataset', fontsize=15)
ax.legend(labels, fontsize=15)

plt.show()

# question 9

wine_data = pd.read_csv("winequalityN.csv")
wine_data['quality'] = (wine_data['quality'] >= 8).astype(int)
features = wine_data.iloc[:, 1:]
target = wine_data['quality']

scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

pca = PCA(n_components=11)
pca_components = pca.fit_transform(standardized_features)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ['pink', 'red']
labels = ['low-quality', 'quality']

pca_8 = pca_components[:, 8]
pca_9 = pca_components[:, 9]

for label_value, color in zip(set(target), colors):
    ax.scatter(
        pca_8[target == label_value],
        pca_9[target == label_value],
        c=color,
        s=60,
        label=labels[label_value]
    )

ax.set_title('PCA of Wine Quality Dataset', fontsize=15)
ax.set_xlabel('Principal Component - 8', fontsize=15)
ax.set_ylabel('Principal Component - 9', fontsize=15)
ax.legend(labels, fontsize=15)

plt.show()
