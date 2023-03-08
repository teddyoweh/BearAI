from BearAI import Tokenizer
from BearAI import Logistic
from BearAI import KNN
from BearAI import Linear


# X_train = ['I love this movie', 'This movie is great', 'The acting is terrible', 'I hate this movie']
# y_train = [1, 1, 0, 0]

# X_test = ['i love cooking fr']

x = [1,2,3,4,5,6,7]
y=[2*i for i in x]
# print(y)

lin = Linear()
lin.fit(x,y)
print(lin.predict(10))
# tokenizer = Tokenizer()
# tokenizer.fit(X_train)
# X_train_transformed = tokenizer.transform(X_train)
# X_test_transformed = tokenizer.transform(X_test)


# logistic = KNN()
# logistic.fit(X_train_transformed, y_train)
# y_pred = logistic.predict(X_test_transformed)

 
# print(y_pred)   





