import cv2
import numpy as np
import joblib

print("================================")
print("   IDENTIFICADOR DE ROSAS 🌹")
print("================================")

modelo = joblib.load("modelo_rosas.pkl")

caminho = input("\nDigite o caminho da foto da rosa: ")

imagem = cv2.imread(caminho)

if imagem is None:
    print("\nNão foi possível abrir essa imagem.")
else:
    imagem = cv2.resize(imagem, (64, 64))
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    histograma = cv2.calcHist(
        [imagem], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )

    histograma = cv2.normalize(histograma, histograma).flatten()
    histograma = np.array([histograma])

    previsao = modelo.predict(histograma)[0]
    probabilidades = modelo.predict_proba(histograma)[0]

    confianca = max(probabilidades) * 100

    nomes = {
        "rosa_negra_halfeti": "Rosa Negra de Halfeti",
        "rosa_azul": "Rosa Azul",
        "rosa_amnesia": "Rosa Amnesia"
    }

    print("\n🌹 ROSA IDENTIFICADA:")
    print(nomes[previsao])
    print(f"Confiança estimada: {confianca:.2f}%")