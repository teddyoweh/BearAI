
import numpy as np
import re
from collections import Counter
class BearAi:
    def __init__():
        pass
    def linear():
        pass


class Linear:
    def __init__(self):
        pass
    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        self.m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        self.b = y_mean - self.m * x_mean
        
    def predict(self,x_test):
        x_test = np.array(x_test)
        y_pred = self.m * x_test + self.b
        
        return y_pred
 


class Logistic:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, x_train, y_train):
   
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1, 1)
        self.m, self.n = x_train.shape
        self.w = np.zeros((self.n, 1))
        self.b = 0
        
 
        for i in range(self.num_iterations):
            z = np.dot(x_train, self.w) + self.b
            y_pred = self.sigmoid(z)
            dw = (1 / self.m) * np.dot(x_train.T, (y_pred - y_train))
            db = (1 / self.m) * np.sum(y_pred - y_train)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, x_test):
        x_test = np.array(x_test)
        z = np.dot(x_test, self.w) + self.b
        y_pred = self.sigmoid(z)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred.ravel().tolist()

class Tokenizer:
    def __init__(self):
        self.word_count = Counter()
    
    def fit(self, X):
        for text in X:
            words = self.tokenize(text)
            self.word_count.update(words)
    
    def tokenize(self, text):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def transform(self, X):
        X_transformed = []
        for text in X:
            words = self.tokenize(text)
            x = [self.word_count[word] for word in words]
            X_transformed.append(x)
        return X_transformed
    



class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        y_pred = []
        X = np.array(X)
        for i in range(X.shape[0]):
            distances = np.sqrt(((self.X_train - X[i])**2).sum(axis=1))
            nearest_indices = distances.argsort()[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            y_pred.append(Counter(nearest_labels).most_common(1)[0][0])
        return np.array(y_pred)
    
# import numpy as np
# from nltk.tokenize import word_tokenize

# class WordSimilarityPredictor:
   
#     def __init__(self, word_embeddings):
      
#         self.word_embeddings = word_embeddings
    
#     def tokenize(self, sentence):
     
#         return Tokenizer(sentence)
#     def cosine_similarity(u, v):
    
#         numerator = np.dot(u, v)
#         denominator = np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))
#         return numerator / denominator
#     def predict_similarity(self, word1, word2):
 
#         if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
#             return None
        
#         embedding1 = self.word_embeddings[word1]
#         embedding2 = self.word_embeddings[word2]
#         return self.cosine_similarity(embedding1, embedding2)

# x_train = [[1, 2], [2, 3], [3, 8], [4, 9]]
# y_train = [0, 0, 1, 1]

# # Define some test data
# x_test = [[5, 6], [6, 11]]

# log = KNN()
# log.fit(x_train, y_train)
# print(log.predict(x_test))
# lin = Linear()
# x = [1,2,3,4,5,6,7,8]
# y = [(2*i)+1 for i in x]
# lin.fit(x,y)
# print(y)
# print(lin.predict(9))