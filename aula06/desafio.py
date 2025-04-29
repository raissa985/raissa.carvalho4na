# Instalação das bibliotecas (execute apenas se não estiverem instaladas)
# !pip install gensim matplotlib scikit-learn pandas numpy spacy plotly

# Importações básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.manifold import TSNE
import re
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download de recursos NLTK (se necessário)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Dados de exemplo - críticas de filmes (simplificadas)
textos = [
    "Este filme é incrível, adorei a atuação do protagonista",
    "A direção de fotografia é espetacular e o roteiro é envolvente",
    "Péssimo filme, desperdicei meu tempo assistindo isso",
    "Os atores são talentosos mas o roteiro é fraco",
    "Cinematografia belíssima, recomendo assistir no cinema",
    "Não gostei da história, personagens mal desenvolvidos",
    "A trilha sonora combina perfeitamente com as cenas",
    "Filme entediante, previsível do início ao fim",
    "Os efeitos especiais são impressionantes, tecnologia de ponta",
    "História emocionante, chorei no final do filme"
]

# Verificando os dados
for i, texto in enumerate(textos[:3]):  # Mostrando apenas os 3 primeiros
    print(f"Texto {i+1}: {texto}")

    from nltk.corpus import stopwords

def preprocessar_texto(texto):
    # Converter para minúsculas
    texto = texto.lower()

    # Remover caracteres especiais e números
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ ]', '', texto)

    # Tokenizar
    tokens = word_tokenize(texto)

    # Remover stopwords (opcional, dependendo da aplicação)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

# Aplicar pré-processamento a todos os textos
textos_preprocessados = [preprocessar_texto(texto) for texto in textos]


# Definir parâmetros do modelo
vector_size = 100    # Dimensionalidade dos vetores
window = 5           # Tamanho da janela de contexto
min_count = 1        # Frequência mínima das palavras
workers = 4          # Número de threads para treinamento
sg = 1               # Modelo Skip-gram (1) ou CBOW (0)

# Treinar o modelo
model = Word2Vec(
    sentences=textos_preprocessados,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    sg=sg
)

print(f"Modelo treinado com {len(model.wv.key_to_index)} palavras no vocabulário")


# Salvar o modelo
model.save("word2vec_filmes.model")

# Carregando o modelo salvo
modelo_carregado = Word2Vec.load("word2vec_filmes.model")
print("Modelo carregado com sucesso!")

# Função para gerar vetores de documento usando embeddings
def texto_para_vetor(texto, modelo):
    """Converte um texto em um vetor médio dos embeddings de suas palavras"""
    palavras = preprocessar_texto(texto)

def similaridade_documentos(doc1, doc2, modelo):
    """Calcula a similaridade entre dois documentos usando embeddings"""
    vetor1 = texto_para_vetor(doc1, modelo)
    vetor2 = texto_para_vetor(doc2, modelo)

    # Calcular similaridade do cosseno
    # similaridade = 1 - distância do cosseno
    similaridade = np.dot(vetor1, vetor2) / (np.linalg.norm(vetor1) * np.linalg.norm(vetor2))
    return similaridade

# Exercício: Calcule a similaridade entre os documentos abaixo
documento1 = "O filme tem uma história envolvente e atuações convincentes"
documento2 = "A narrativa do filme é cativante e os atores são excelentes"
documento3 = "O restaurante tem comida deliciosa e preços acessíveis"

# Calcular similaridades (implemente sua solução)
