import cv2
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Caminho das imagens
PASTA_IMAGENS = "imagens"

# Categorias das rosas
CATEGORIAS = [
    "rosa_negra_halfeti",
    "rosa_azul",
    "rosa_amnesia"
]

imagens = []
rotulos = []

print("Carregando imagens...")

# Ler as imagens de cada categoria
for categoria in CATEGORIAS:
    caminho = os.path.join(PASTA_IMAGENS, categoria)

    for arquivo in os.listdir(caminho):
        caminho_imagem = os.path.join(caminho, arquivo)

        imagem = cv2.imread(caminho_imagem)

        if imagem is not None:
            imagem = cv2.resize(imagem, (64, 64))
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

            histograma = cv2.calcHist(
                [imagem], [0, 1, 2], None,
                [8, 8, 8], [0, 180, 0, 256, 0, 256]
            )

            histograma = cv2.normalize(histograma, histograma).flatten()

            imagens.append(histograma)
            rotulos.append(categoria)

# Preparar os dados
X = np.array(imagens)
y = np.array(rotulos)

print(f"\nTotal de imagens carregadas: {len(X)}")

# Separar dados para treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Criar e treinar o modelo
modelo = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", probability=True, random_state=42)
)

print("\nTreinando o modelo...")
modelo.fit(X_treino, y_treino)

# Avaliar o resultado
previsoes = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, previsoes)

print("\n===== RESULTADO DO TREINAMENTO =====")
print(f"Acurácia: {acuracia * 100:.2f}%")

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes, zero_division=0))
joblib.dump(modelo, "modelo_rosas.pkl")
print("Modelo salvo com sucesso!")
print("\nModelo treinado com sucesso!")