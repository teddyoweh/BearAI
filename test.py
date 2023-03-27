# from BearAI import CosineSimilarity,Tokenizer


# file1 = open('Node.py','r').read()
# file2 = open('Node1.py','r').read()




# csn = CosineSimilarity(file1, file2)
# print(csn.value)
# print(csn.get_similar_words(0.6))
import pandas as pd

class MultiTable:

    def __init__(self):
   
        self.tables={}
    @property
    def table_names(self):
       return list(self.tables.keys())
    def add(self,table):
        self.tables[table.table_name]=table
class Table:
    def __init__(self):
        self.headers:list[str]
        self.column_data:list
      
    @property
    def total_data(self):
      return {self.headers[i]:sum(self.column_data[i]) for i in range(1,len(self.headers))}
    @property
    def to_df(self):
        rows = []
        temp=[]
        for i in range(len(self.column_data[0])):
            for j in range(len(self.column_data)):
        
                temp.append(self.column_data[j][i])
            rows.append(temp)
            temp = []

        return pd.DataFrame(rows,columns=self.headers)
    
    def P2(self,head,target):
      head ,target= head.lower(),target.lower()
      df = self.to_df
      return list(df[df[self.headers[0]]==head][target])[0]/self.total_data[target]
    
    def P1(self,var,t=False):
      df = self.to_df
      if(t==False):
        return {'ans':self.total_data[var]/sum(self.total_data.values()),'frac':f'{self.total_data[var]}/{sum(self.total_data.values())}'}
      else:
        return {'ans':list(df[df[self.headers[0]]==var].iloc[0][1::])[0]/sum(self.total_data.values()),'frac':f'{list(df[df[self.headers[0]]==var].iloc[0][1::])[0]}/{sum(self.total_data.values())}'}
    def calc(self,A,B):
        return (self.P2(A,B)*self.P1(B))/self.P1(A,t=True)

      
class Freq(MultiTable):
   def __init__(self,df,target_label,dfis=False ) -> None:
    super().__init__()
    if(dfis):
       df=df
    else:
        df = pd.read_csv(df)

    targets = set(df[target_label])

    for _ in df.drop(columns=['Play']).columns:
        ftable = Table()
        ftable.headers = [_,*targets]
        ftable.table_name = _
        uqn = set(df[_])
        ftable.column_data = [[*list(uqn)]]
        for j in targets:
            temp = []
            for i in uqn:
                te =df[df[_]==i]
                temp.append(len(te[df[target_label]==j]))
            ftable.column_data.append(temp)
            self.add(ftable)


# print(Freq('data.csv','Play').tables['Outlook'].to_df)

    

