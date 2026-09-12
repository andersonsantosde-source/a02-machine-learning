from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Previsão Churn", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
        .stAlert, .stProgress > div {
            border-radius: 14px;
        }
        .card {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_modelo():
    caminho = Path(__file__).resolve().parent / "modelo_churn_v1.pkl"
    if not caminho.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute a etapa 1 antes de abrir a interface.")
    return joblib.load(caminho)


FEATURES = [
    "tempo_contrato",
    "valor_mensal",
    "reclamacoes",
    "nota_satisfacao",
    "chamados_suporte",
]

st.title("Previsão de Churn")
st.caption("Analise o risco de cancelamento de clientes com inteligência artificial.")

modelo = carregar_modelo()

opcao = st.radio(
    "Selecione o tipo de análise:",
    ["Análise Individual", "Análise em Lote"],
    horizontal=True,
)

if opcao == "Análise Individual":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("form_cliente"):
        col1, col2 = st.columns(2)

        with col1:
            tempo_contrato = st.slider("Tempo de contrato (meses)", 0, 120, 24)
            valor_mensal = st.number_input(
                "Valor mensal",
                min_value=0.0,
                max_value=5000.0,
                value=95.0,
                step=1.0,
            )
            reclamacoes = st.slider("Número de reclamações", 0, 10, 1)

        with col2:
            nota_satisfacao = st.slider("Nota de satisfação", 1, 5, 4)
            chamados_suporte = st.slider("Chamados de suporte", 0, 20, 1)

        enviado = st.form_submit_button("Prever churn", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if enviado:
        dados_cliente = pd.DataFrame(
            [{
                "tempo_contrato": tempo_contrato,
                "valor_mensal": valor_mensal,
                "reclamacoes": reclamacoes,
                "nota_satisfacao": nota_satisfacao,
                "chamados_suporte": chamados_suporte,
            }],
            columns=FEATURES,
        )

        previsao = modelo.predict(dados_cliente)[0]
        probabilidade = modelo.predict_proba(dados_cliente)[0, 1]

        st.subheader("Resultado da previsão")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Probabilidade de churn", f"{probabilidade:.1%}")

        with col2:
            status = "Alto risco" if previsao == 1 else "Baixo risco"
            st.metric("Status do cliente", status)

        if previsao == 1:
            st.error(f"O cliente tem alto risco de churn. Probabilidade estimada: {probabilidade:.1%}")
            st.warning("Atenção: recomenda-se ação de retenção com foco em prevenção de cancelamento.")
        else:
            st.success(f"O cliente tem baixo risco de churn. Probabilidade estimada: {probabilidade:.1%}")
            st.info("Situação estável: cliente com baixa probabilidade de cancelamento.")

        st.progress(float(probabilidade))

else:
    st.subheader("Análise em Lote")
    uploaded_file = st.file_uploader(
        "Faça upload de um CSV com as colunas do cliente",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            dados_lote = pd.read_csv(uploaded_file)
            colunas_faltantes = [coluna for coluna in FEATURES if coluna not in dados_lote.columns]

            if colunas_faltantes:
                st.error(f"Arquivo sem as colunas necessárias: {colunas_faltantes}")
            else:
                dados_lote = dados_lote[FEATURES].copy()
                probabilidade = modelo.predict_proba(dados_lote)[:, 1]
                previsao = modelo.predict(dados_lote)

                resultado = dados_lote.copy()
                resultado["probabilidade_churn"] = probabilidade
                resultado["previsao_churn"] = previsao
                resultado["classificacao"] = resultado["previsao_churn"].map({0: "Baixo risco", 1: "Alto risco"})

                st.success(f"{len(resultado)} clientes processados com sucesso.")
                st.dataframe(resultado, use_container_width=True)

                contagem = resultado["classificacao"].value_counts().rename_axis("status").reset_index(name="quantidade")
                st.bar_chart(contagem.set_index("status")["quantidade"])

        except Exception as erro:
            st.exception(f"Erro ao processar o arquivo: {erro}")
    else:
        st.info("Envie um CSV com as colunas: tempo_contrato, valor_mensal, reclamacoes, nota_satisfacao, chamados_suporte.")

st.markdown("---")
st.subheader("Como a previsão é feita")
st.write(
    "A aplicação usa o modelo treinado com os dados históricos do cliente e analisa as variáveis "
    "mais relevantes para estimar a chance de cancelamento."
)


