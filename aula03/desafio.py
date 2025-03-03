#Neste código, usaremos um modelo de Machine Learning chamado KNN (K-Nearest Neighbors) para aprender a diferenciar três tipos de vinho com 
#base em dados pré-existentes e depois testaremos a precisão do modelo.

#Importando as bibliotecas necessárias
import pandas as pd  #Para manipulação de dados em tabelas
from sklearn.model_selection import train_test_split  #Para dividir os dados em treino e teste
from sklearn.preprocessing import StandardScaler  #Para normalizar os dados (deixar tudo na mesma escala)
from sklearn.neighbors import KNeighborsClassifier  #Algoritmo de aprendizado de máquina KNN (K-Nearest Neighbors)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score  #Métricas de avaliação
from sklearn.datasets import load_wine  #Conjunto de dados de vinhos pré-pronto

#Carregar o conjunto de dados Wine
data = load_wine() #Dataset que contém informações sobre diferentes tipos de vinho
wine_df = pd.DataFrame(data.data, columns=data.feature_names)#Cria uma tabela com os dados dos vinhos
y = data.target #Pega a categoria de cada vinho

#Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(wine_df, y, test_size=0.3, random_state=42)
#Aloca 70% dos dados para treinar o modelo e 30% psra testar o modelo
#A opção 'random_state=42' garante que os resultados sejam sempre os mesmos ao rodar o código novamente

#Normalizar os dados
scaler = StandardScaler() #Criado um normalizador para padronizar os dados
X_train = scaler.fit_transform(X_train) #Ajusta o normalizador com os dados de treino e transforma os valores
X_test = scaler.transform(X_test) #Aplica a mesma transformação nos dados de teste
#Muito importante porque algumas colunas possuem valores muito diferentes entre si e normalizar ajuda o modelo a aprender melhor

#Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5) #Cria um modelo KNN com 5 vizinhos
knn.fit(X_train, y_train) #Treina o modelo usando os dados de treino

#Fazer previsões
y_pred = knn.predict(X_test) #Usa o modelo treinado para prever as categorias dos vinhos de teste

#Calcular e exibir métricas
test_metrics = {
    "Acurácia": accuracy_score(y_test, y_pred), #Mede quantas previsões o modelo acertou
    "Matriz de Confusão": confusion_matrix(y_test, y_pred), #Mostra erros e acertos por categoria
    "Precisão": precision_score(y_test, y_pred, average='weighted'),#Mede a qualidade das previsões corretas
    "Recall": recall_score(y_test, y_pred, average='weighted'), #Mede quantas amostras de cada categoria foram corretamente identificadas
    "F1-Score": f1_score(y_test, y_pred, average='weighted') #Combina precisão e recall para avaliar o desempenho geral
}
#Obs: o 'average=weighted' é usado porque temos múltiplas categorias de vinho.

#Exibir os resultados
for metric, value in test_metrics.items():
    print(f"{metric}:\n{value}\n")