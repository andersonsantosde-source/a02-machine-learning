#Etapa 1: importando módulos necessários
import pandas as pd #ferramenta para trabalhar com tabelas
import joblib #exporta o modelo

#scikit-learn: vamos pegar módulos específicos
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import seaborn as sns 

#Etapa 2: Criar um dicionário de dados a serem analisados
try:
    print("Carregando arquivo 'churn_data.csv'...")
    df = pd.read_csv('churn_data.csv') #lê o arquivo 
    print(f"Sucesso! {len(df)} linhas importadas.")

except FileNotFoundError:
    print("Erro: arquivo 'churn_data.csv não foi encontrado.")
    exit()

#Etapa 3: Pré-processamento de dados
#vamos separar as perguntas das respostas, nesse caso nossa pergunta é:o cliente cancelou?
#X = tudo menos a coluna Cancelou, são as pistas que a IA vai olhar
X = df.drop('cancelou', axis=1)
#y = apenas a coluna 'cancelou'. É o que queremos que o modelo aprenda a prever
y = df['cancelou']

#Dividindo treino [80%] e teste [20%]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)
print("\n Dados divididos de forma estratificada.")

#Etapa 4: treinamento do modelo e previsão de dados
modelo_churn = Pipeline ([ #Pipeline: agrupa as tarefas em sequência
    ('padronizador', StandardScaler()), #padroniza todos os dados numa escala de 0 a 100
    ('classificador', RandomForestClassifier(  #algoritmo de classificação que combina o palpite de várias árvores
        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        #n_estimators = quantidade de árvores criadas
        #min_samples_leaf = quantidade de exemplos/testes por árvore
        #random_state = trava o gerador de números aleatórios
        #n_jobs = diz pra máquina que está processando utilizar todos os núcleos
    )),
])

#comando fit: treina e ajusta o modelo
#aqui a IA está comparando os dados das perguntas (X) com as respostas (y)
modelo_churn.fit(X_train,y_train)

#comando predict: previsão das informações/
previsoes = modelo_churn.predict(X_test) #prevê se o cliente irá churnar
probabilidades = modelo_churn.predict_proba(X_test)[:,1] #extrair a probabilidade 
#individual de cada cliente cancelar com base nos dados de treino

#Etapa 5: avaliação do modelo 
print("\n### Relatório de performance ###")
print(classification_report(y_test, previsoes))
print(f"ROC AUC: {roc_auc_score(y_test,probabilidades):.3f}")

#Etapa 6: deploy (salvando o modelo)
joblib.dump(modelo_churn,'modelo_churn_v1.pkl')
print("Modelo foi exportado com sucesso.")
















