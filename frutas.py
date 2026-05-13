import pandas as pd
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#fazendo a leitura dos meus dados
df = pd.read_excel("data/frutas.xlsx")

print(df)
df.duplicated().sum()
df = df.drop_duplicates()

#separando o modelo

#reposta
y = df['Fruta']

#Caracteristicas
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]


# NORMALIZAÇÃO
scaler = MinMaxScaler()

x_normalizado = scaler.fit_transform(x)

#criando o meu algoritimo
arvore = tree.DecisionTreeClassifier()

#ensinando o algoritimo / ajuste do modelo / treinando os padroes

arvore.fit(x_normalizado, y)

print(arvore.predict([[1,1,1,1]]))


#criação de uma arvore 

plt.figure(dpi=400, figsize=(3,8))
tree.plot_tree(
    arvore,
    feature_names=caracteristicas,
    class_names=arvore.classes_,
    filled=True
)
plt.show()

# trabalhando com probabilidade

probabilidade = arvore.predict_proba([[0,1,0,1]])[0]
pd.Series(probabilidade, index= arvore.classes_)

print(probabilidade)

# ➡️ você não sabe se ele realmente aprendeu padrões ou apenas memorizou os exemplos.
# Isso se chama overfitting. o codigo acima.