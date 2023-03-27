import math

class NaiveBayes:
    def __init__(self):
        self.freq_tables = {}
        self.labels = []

    def train(self, X, y):
 
        for i in range(len(X[0])):
            table = {}
            for j in range(len(X)):
                if X[j][i] not in table:
                    table[X[j][i]] = [0] * len(set(y))
                table[X[j][i]][self.labels.index(y[j])] += 1
            self.freq_tables[i] = table

 
        self.label_probs = {}
        for label in set(y):
            self.label_probs[label] = y.count(label) / len(y)

    def predict(self, X):
        y_pred = []
        for x in X:
            probs = {}
            for label in self.labels:
                prob = self.label_probs[label]
                for i in range(len(x)):
                    if x[i] in self.freq_tables[i]:
                        count = self.freq_tables[i][x[i]][self.labels.index(label)]
                        prob *= count / sum(self.freq_tables[i][x[i]])
                probs[label] = prob
            y_pred.append(max(probs, key=probs.get))
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        correct = sum([1 if y_pred[i] == y[i] else 0 for i in range(len(y))])
        return correct / len(y)
