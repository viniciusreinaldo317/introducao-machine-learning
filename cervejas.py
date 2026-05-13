import pandas as pd
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_excel("data/cervejas.xlsx")
print(df)

# pré-processamento dos dados
df.duplicated().sum()
df = df.drop_duplicates()

# separação da resposta e teste

y = df['Classe']

caracteristicas = ['Temperatura', 'Copo', 'Espuma', 'Cor']
x = df[caracteristicas]
print(x)

# a biblioteca sklearn trabalha com numeração e a nossa base 
# trabalha com string, logo, devemos transforma-los em numero e realizar uma normalização

x = x.replace({
    "mud": 1, "pint": 2,
    "sim" : 1, "nao": 0,
    "clara" : 0, "escura": 1,
})

print(x)

# criando nosso modelo

model = tree.DecisionTreeClassifier(random_state=42)

model.fit(x,y)

# criando uma arvore ilustrativa

plt.figure(dpi=400, figsize=(1,8))
tree.plot_tree(
    model,
    feature_names=caracteristicas,
    class_names=model.classes_,
    filled=True
)

plt.show()