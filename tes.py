from test import Freq
class NaiveBayes:
    def __init__(self, freq_tables):
        self.freq_tables = freq_tables
        
    def predict(self, data_point):
        max_prob = 0
        predicted_label = None
        
        for label, freq_table in self.freq_tables.items():
            prob = 1
            for feature, value in data_point.items():
                prob *= freq_table.calc(value, feature)
            prob *= freq_table.P1(label)['ans']
            
            if prob > max_prob:
                max_prob = prob
                predicted_label = label
                
        return predicted_label

# Sample dataset
data = [
    {'outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'No'},
    {'outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'No'},
    {'outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'No'},
    {'outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'Yes'},
    {'outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'Play': 'No'},
    {'outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play': 'Yes'},
    {'outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'Yes'},
    {'outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'},
    {'outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play': 'No'}
]
import pandas as pd
df= pd.DataFrame(data)
# Create frequency tables for each feature
freq_tables = {}
for feature in ['outlook', 'Temperature', 'Humidity', 'Wind']:
    freq_table = Freq(df, 'Play',True).tables[feature]
    freq_tables[feature] = freq_table

# Create NaiveBayes instance
nb = NaiveBayes(freq_tables)

# Test on sample data
data_point = {'outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
predicted_label = nb.predict(data_point)
print(predicted_label)  # Expected output: 'No'

# data_point = {'outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
# predicted_label = nb.predict(data_point)
# print(predicted_label)  # Expected output: 'Yes'
