import streamlit as st
import cv2
import numpy as np
import joblib

st.set_page_config(
    page_title="Classificador de Rosas",
    page_icon="🌹",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; color:#8B1E3F;'>🌹 Classificador de Rosas</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Descubra a beleza e a variedade das rosas com Inteligência Artificial.</p>",
    unsafe_allow_html=True
)

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_rosas.pkl")

modelo = carregar_modelo()

nomes = {
    "rosa_negra_halfeti": "Rosa Negra de Halfeti",
    "rosa_azul": "Rosa Azul",
    "rosa_amnesia": "Rosa Amnesia"
}

st.markdown(
    "<h3 style='text-align:center; color:#8B1E3F;'>📸 Envie sua fotografia</h3>",
    unsafe_allow_html=True
)

arquivo = st.file_uploader(
    "Selecione uma imagem da rosa",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceitos: JPG, JPEG e PNG."
)

if arquivo is not None:
    dados = np.frombuffer(arquivo.read(), np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)

    if imagem is None:
        st.error("Não foi possível abrir essa imagem.")
    else:
        st.image(
            cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB),
            caption="Foto selecionada",
            use_container_width=True
        )

        imagem_processada = cv2.resize(imagem, (64, 64))
        imagem_processada = cv2.cvtColor(
            imagem_processada,
            cv2.COLOR_BGR2HSV
        )

        histograma = cv2.calcHist(
            [imagem_processada],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        histograma = cv2.normalize(
            histograma,
            histograma
        ).flatten()

        histograma = np.array([histograma])

        previsao = modelo.predict(histograma)[0]
        probabilidades = modelo.predict_proba(histograma)[0]
        confianca = max(probabilidades) * 100

        st.markdown(
    f"""
    <div style="
        background-color: #FCE4EC;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #E8A0B5;
    ">
        <h2 style="color: #8B1E3F;">🌹 Resultado da análise</h2>
        <h1 style="color: #8B1E3F;">{nomes[previsao]}</h1>
    </div>
    """,
    unsafe_allow_html=True
)
        st.markdown(
    f"<h3 style='text-align:center; color:#8B1E3F;'>Confiança estimada: {confianca:.2f}%</h3>",
    unsafe_allow_html=True
)

        st.caption(
            "Resultado experimental. A precisão depende da qualidade "
            "e da variedade das imagens utilizadas no treinamento."
        )
        st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8B1E3F;">
        <h4>🌹 Classificador de Rosas</h4>
        <p>Projeto de Machine Learning e Visão Computacional</p>
        <p>Desenvolvido em Python com OpenCV, Scikit-learn e Streamlit.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    