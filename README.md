# Previsão de Churn

Projeto de machine learning para prever risco de churn de clientes.

## Estrutura

- `interface.py`: interface em Streamlit para análise individual.
- `model_creator.py`: treinamento do modelo e exportação do arquivo `.pkl`.
- `churn_data.csv`: base de dados utilizada para treinamento.
- `modelo_churn_v1.pkl`: modelo treinado exportado.
- `requirements.txt`: dependências do projeto.

## Como executar

1. Crie o ambiente virtual:
   ```bash
   python -m venv .venv
   ```
2. Ative o ambiente:
   ```bash
   .venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Treine o modelo:
   ```bash
   python model_creator.py
   ```
5. Rode a interface:
   ```bash
   streamlit run interface.py
   ```

## Observações

O projeto usa um modelo de Random Forest com pré-processamento por StandardScaler para estimar a chance de cancelamento do cliente.
