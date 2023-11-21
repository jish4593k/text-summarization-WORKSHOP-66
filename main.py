
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Uncomment the lines below if nltk.download() is required
# nltk.download('punkt')   # one-time execution
# nltk.download('stopwords')  # one-time execution


text = "Your input text goes here."

sentence = sent_tokenize(text)


corpus = []
for i in range(len(sentence)):
    sen = re.sub('[^a-zA-Z]', " ", sentence[i])
    sen = sen.lower()
    sen = sen.split()
    sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
    corpus.append(sen)

n = 300
all_words = [i.split() for i in corpus]
model = Word2Vec(all_words, min_count=1, size=n)


sen_vector = [np.mean([model.wv[j] for j in i.split()], axis=0) for i in corpus]

# Converting sentence vectors to PyTorch tensors
sen_vector_tensor = torch.tensor(sen_vector, dtype=torch.float32)

n_clusters = int(input("Number of clusters: "))
kmeans_torch = nn.Sequential(nn.Linear(n, n_clusters), nn.Softmax(dim=1))
optimizer = optim.Adam(kmeans_torch.parameters(), lr=0.01)


epochs = 100
for epoch in range(epochs):
    outputs = kmeans_torch(sen_vector_tensor)
    loss = nn.KLDivLoss()(outputs.log(), outputs.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


_, y_kmeans_torch = torch.max(outputs, 1)


my_list_torch = []
for i in range(n_clusters):
    cluster_indices = torch.where(y_kmeans_torch == i)[0]
    distances = [distance.euclidean(sen_vector[j], kmeans_torch[0].weight[i].detach().numpy()) for j in cluster_indices]
    min_distance_index = cluster_indices[np.argmin(distances)]
    my_list_torch.append(min_distance_index)

print("Indices of sentences closest to cluster centroids (PyTorch):", my_list_torch)
print("Assigned clusters for each sentence (PyTorch):", y_kmeans_torch.numpy())

for i in sorted(my_list_torch):
    print(sentence[i])

plt.scatter(sen_vector_tensor.numpy()[:, 0], sen_vector_tensor.numpy()[:, 1], c=y_kmeans_torch.numpy(), cmap='viridis', alpha=0.5, s=50)
plt.scatter(kmeans_torch[0].weight.detach().numpy()[:, 0], kmeans_torch[0].weight.detach().numpy()[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering of Sentences (PyTorch)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
