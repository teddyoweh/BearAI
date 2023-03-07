from BearAI import Tokenizer
from BearAI import Logistic
from BearAI import KNN

X_train = ['I love this movie', 'This movie is great', 'The acting is terrible', 'I hate this movie']
y_train = [1, 1, 0, 0]

X_test = [' movie fr']

 
tokenizer = Tokenizer()
tokenizer.fit(X_train)
X_train_transformed = tokenizer.transform(X_train)
X_test_transformed = tokenizer.transform(X_test)


logistic = KNN()
logistic.fit(X_train_transformed, y_train)
y_pred = logistic.predict(X_test_transformed)

 
print(y_pred)   





