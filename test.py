from BearAI import CosineSimilarity,Tokenizer


file1 = open('Node.py','r').read()
file2 = open('Node1.py','r').read()




csn = CosineSimilarity(file1, file2)
print(csn.value)
print(csn.get_similar_words(0.6))