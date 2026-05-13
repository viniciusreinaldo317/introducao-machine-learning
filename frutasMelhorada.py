import pandas as pd
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# leitura
df = pd.read_excel("data/frutas.xlsx")

# removendo duplicados
df = df.drop_duplicates()

# resposta
y = df['Fruta']

# características
caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]

# separação treino/teste
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# normalização
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# modelo
arvore = tree.DecisionTreeClassifier()

# treino
arvore.fit(x_train, y_train)

# previsões
resultado = arvore.predict(x_test)

# precisão
precisao = accuracy_score(y_test, resultado)

print("Precisão:", precisao)