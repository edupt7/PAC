import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from scipy import stats

from scipy.fft import fft
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

#variaveis globais
count_error=0
min_amostras=15
janela_tempo=100
modelo = None
modelo_rf = None
classes=None

def autocorrelacao(sinal):
    
    n = len(sinal)
    mean = np.mean(sinal)
    var = np.var(sinal)
    
    if var == 0:  # Se a variância for zero, retorne zero (sem variabilidade nos dados)
        return 0
    
    acf = np.correlate(sinal - mean, sinal - mean, mode='full') / var / n
    return acf[n-1]  # Autocorrelação no lag 0

def entropia_shannon(sinal):
    """Calcula a entropia de Shannon de um sinal"""
    probabilidade = np.histogram(sinal, bins=50, density=True)[0]
    probabilidade = probabilidade[probabilidade > 0]
    return -np.sum(probabilidade * np.log2(probabilidade))

def mav(sinal):
    """Calcula o valor absoluto médio (Mean Absolute Value)"""
    return np.mean(np.abs(sinal))

def rms(sinal):
    """Calcula a raiz quadrada da média dos quadrados (Root Mean Square)"""
    return np.sqrt(np.mean(sinal**2))

def distancia_euclidiana(amostras):
    """Calcula a distância euclidiana entre pontos consecutivos"""
    dist = np.sqrt(np.sum(np.diff(amostras, axis=0)**2, axis=1))
    return np.mean(dist)  # Retorna a média das distâncias

def amplitude(amostras):
    """Calcula a amplitude do sinal (diferença entre o máximo e o mínimo)"""
    return np.max(amostras) - np.min(amostras)

def variancia(sinal):
    """Calcula a variância"""
    return np.var(sinal)

def poder_espectral(sinal):
    """Calcula o poder espectral de um sinal usando FFT"""
    sinal_fft = fft(sinal)
    return np.sum(np.abs(sinal_fft)**2)  # Soma dos quadrados dos coeficientes

def correlacao_cruzada(amostras):
    """Calcula a correlação cruzada entre os eixos (X, Y, Z)"""
    # Verifica se a variância de cada eixo é zero antes de calcular a correlação
    if np.var(amostras[:, 0]) == 0 or np.var(amostras[:, 1]) == 0 or np.var(amostras[:, 2]) == 0:
        return np.zeros((3, 3))  # Retorna uma matriz de correlação nula
    
    return np.corrcoef(amostras.T)  # Matriz de correlação entre as colunas (eixos X, Y, Z)

def leituraDados(fileName,dados=[]):
    global count_error
    global min_amostras

    if not os.path.isabs(fileName):
        name = os.path.join(os.path.dirname(__file__),'data','test',fileName)
    else:
        name=fileName


    try:
        with open(name, 'r') as file:
            # Lê as primeiras 11 linhas
            for i in range(11):
                line = file.readline()  # Lê linha por linha
                if i==3:
                    parts = line.strip().split(",")
                    if len(parts) > 1:  # Verifica se existe pelo menos 2 elementos
                        idade = parts[1].strip()  # Extrai o valor da atividade 
                if i==4:
                    parts = line.strip().split(",")
                    if len(parts) > 1:  # Verifica se existe pelo menos 2 elementos
                        altura = parts[1].strip()  # Extrai o valor da atividade 
                
                if i==5:
                    parts = line.strip().split(",")
                    if len(parts) > 1:  # Verifica se existe pelo menos 2 elementos
                        peso = parts[1].strip()  # Extrai o valor da atividade 
                
                if i==6:
                    parts = line.strip().split(",")
                    if len(parts) > 1:  # Verifica se existe pelo menos 2 elementos
                        sexo = parts[1].strip()  # Extrai o valor da atividade 
                     
                if i == 7:  # Linha 8 (index 7) contém a atividade
                    parts = line.strip().split(",")
                    if len(parts) > 1:  # Verifica se existe pelo menos 2 elementos
                        atividade = parts[1].strip()  # Extrai o valor da atividade
                        if atividade == "Cair para a direita" or atividade == "Cair para a esquerda" or atividade == "Cair para a frente" or atividade == "Cair para tras":
                            atividade ="Queda"
                        elif atividade =="Levantar depois de Deitar" or atividade =="Levantar depois de Sentar":
                            atividade ="Levantar"
                        elif atividade =="Caminhar" or atividade=="Correr":
                            atividade="Movimento"
                    else:
                        atividade = "Desconhecida"  # Valor default se não houver atividade
                

        #lê todas as 3 primeiras colunas so apartir da 11 linha X,Y,Z

        df = pd.read_csv(name, skiprows=11, header=None)  
        dados = df[[0, 1, 2]].values.tolist()

       
        return [idade,altura,peso,sexo,atividade, dados]

       

    except Exception as e:
        count_error +=1
        return None  # Retorna None para arquivos com erro

def leituraTodosDados(dados=[]):
    global count_error
    for file in ['test','train','validation']:
        dir = os.path.join(os.path.dirname(__file__), 'data', file) 
        arquivos = os.listdir(dir)
        arquivos_csv = [arq for arq in arquivos if arq.endswith('.csv')]


        
        for arquivo in arquivos_csv:
            caminho_arquivo = os.path.join(dir, arquivo)
            resultado = leituraDados(caminho_arquivo)

            if resultado:  # Só adiciona os arquivos que foram lidos corretamente
                idade,altura,peso,sexo,actividade, dados_arquivo = resultado
                dados.append([actividade, file, dados_arquivo,idade, altura, peso, sexo])
            else:
                count_error +=1
                
    return dados    

def printDados(dados):
    global min_amostras
    atividades = ["Levantar", "Deitar", "Sentar", "Queda", "Movimento"]
    contador = {
        "test": {atividade: 0 for atividade in atividades},
        "validation": {atividade: 0 for atividade in atividades},
        "train": {atividade: 0 for atividade in atividades}
    }
    test=train=validation=0
    Stest=Strain=Svalidation=0
   
    for n, (actividade, tipo, amostras,idade, altura, peso, sexo) in enumerate(dados):    
        # Contagem para o tipo de dado
        if len(amostras)>min_amostras:
            if tipo == "test":
                test += 1
                Stest += len(amostras)
                contador["test"][actividade] += 1
            elif tipo == "validation":
                validation += 1
                Svalidation += len(amostras)
                contador["validation"][actividade] += 1
            elif tipo == "train":
                train += 1
                Strain += len(amostras)
                contador["train"][actividade] += 1


    data = []
    for tipo in ['train', 'test', 'validation']:
        for atividade in atividades:
            data.append({
                "Tipo": tipo.capitalize(),
                "Atividade": atividade,
                "Quantidade": contador[tipo][atividade]
            })

    df = pd.DataFrame(data)

    # Gráfico de barras empilhadas com Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Tipo", y="Quantidade", hue="Atividade", data=df, palette="Set2")
    plt.title("Distribuição das Atividades por Tipo de Dados")
    plt.xlabel("Conjunto de Dados")
    plt.ylabel("Quantidade de Amostras")
    plt.legend(title="Atividade")
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()
    
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"|{'Tipo':^20s}|{'Media de amostras':^25s}|{'Quantidade de dados':^25s}|{'Queda':^10s}|{'Movimento':^15s}|{'Levantar':^15s}|{'Deitar':^10s}|{'Sentar':^10s}|")

    for tipo in ['train', 'test', 'validation']:
        print(f"|{tipo:^20s}|{(eval(f'S{tipo}')/eval(tipo)):^25.1f}|{eval(tipo):^25.0f}|"
            f"{contador[tipo]['Queda']:^10d}|{contador[tipo]['Movimento']:^15d}|"
            f"{contador[tipo]['Levantar']:^15d}|{contador[tipo]['Deitar']:^10d}|"
            f"{contador[tipo]['Sentar']:^10d}|")
        
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
        
def prepararDadosParaCNN(dados):
    global janela_tempo
    global min_amostras
    global classes

    input = []  # Lista para as entradas
    out = []  # Lista para as (atividades)

    # Preparar os dados para entrada na CNN
    for actividade, tipo, amostras,idade, altura, peso, sexo in dados:

        if len(amostras)> min_amostras:
            if len(amostras) < janela_tempo:
                n_amostras = len(amostras)


                # Verificar se já há dados suficientes para calcular a média dos últimos 5 elementos
                if n_amostras >= 5:
                    media_ultimos_5 = np.mean(amostras[-5:], axis=0)   # Calcular a média dos últimos 5 elementos
                else:
                    media_ultimos_5 = np.mean(amostras, axis=0)          # Se não houver dados suficientes, use a média de todos os elementos disponíveis

                # Criar o padding com a média dos últimos 5 elementos
                padding = np.tile(media_ultimos_5, (janela_tempo - n_amostras, 1))  # Replicar para o número necessário de linhas   
                amostras = np.vstack([amostras, padding])  # Adicionar o padding à lista amostras
                
            elif len(amostras) > janela_tempo:           
                maxj=round(len(amostras)/2+janela_tempo/2)
                minj=round(len(amostras)/2-janela_tempo/2)
                
                amostras = amostras[minj:maxj:] 

            
           
            #scaler =StandardScaler()
            # Normaliza cada coluna (X, Y, Z) independentemente
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaler_z = MinMaxScaler()
            amostras = np.array(amostras)

            # Normalizar independentemente cada eixo
            X_norm = scaler_x.fit_transform(amostras[:, 0].reshape(-1, 1))
            Y_norm = scaler_y.fit_transform(amostras[:, 1].reshape(-1, 1))
            Z_norm = scaler_z.fit_transform(amostras[:, 2].reshape(-1, 1))

            # Recombinar as colunas normalizadas
            amostras = np.hstack([X_norm, Y_norm, Z_norm])

            
            
            amostras = np.array(amostras)
            input.append(amostras)
            out.append(actividade)

    # Converter listas para arrays numpy
    input = np.array(input)
    encoder = LabelEncoder()
    out = encoder.fit_transform(out)
    out = np.array(out)
    classes=encoder.classes_

    return input, out

def divisaoDados(dados):
    train = [line for line in dados if line[1] == 'train']
    test = [line for line in dados if line[1] == 'test']
    validation = [line for line in dados if line[1] == 'validation']
    return train,test,validation

def criarModelo(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),   #filtros
        MaxPooling1D(2),
        Flatten(),
        #Dropout(0.2),    #de maneira aleatoria ele desliga uma parte dos "neuronios" em cada camada para nao depender de neuronios especificos
        Dense(128, activation='relu'),  #neuronios
        #Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)), #força a que detete padroes mais simples
        Dense(num_classes, activation='softmax')  # Última camada com softmax para classificação
        #Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def modeloCNN(dados):
    global modelo
    dados_train,dados_test,dados_validation=divisaoDados(dados)
    X_train,Y_train=prepararDadosParaCNN(dados_train)
    X_test,Y_test=prepararDadosParaCNN(dados_test)
    X_val,Y_val=prepararDadosParaCNN(dados_validation)
  
    input_shape = (X_train.shape[1], X_train.shape[2])  # (janela_tempo, número de features)
    num_classes = len(np.unique(Y_train))  # Número de classes distintas
    modelo = criarModelo(input_shape, num_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)   #ele interrompe o modelo quando au longo de 5 temporadas ele nao melhora
    history=modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, callbacks=[early_stopping])
    loss, accuracy = modelo.evaluate(X_val, Y_val)
    print(f"Perda na validaçao: {loss:.4f}, Precisão na validaçao: {accuracy:.4f}")

def extrair_caracteristicas(dados):
    X, Y, Z = dados[:, 0], dados[:, 1], dados[:, 2]  # Divida os eixos X, Y, Z
    
    media = np.mean(dados, axis=0)
    desvio = np.std(dados, axis=0)
    maximo = np.max(dados, axis=0)
    minimo = np.min(dados, axis=0)
    energia = np.sum(dados**2, axis=0)
    skewness = stats.skew(dados, axis=0)
    curtose = stats.kurtosis(dados, axis=0)

   # Novas características
    acf_x = autocorrelacao(X)
    acf_y = autocorrelacao(Y)
    acf_z = autocorrelacao(Z)
    
    mav_x = mav(X)
    mav_y = mav(Y)
    mav_z = mav(Z)
    
    rms_x = rms(X)
    rms_y = rms(Y)
    rms_z = rms(Z)
    
    
    poder_x = poder_espectral(X)
    poder_y = poder_espectral(Y)
    poder_z = poder_espectral(Z)
    
    # Correlação cruzada entre os eixos
    corr_xy = correlacao_cruzada(dados)[0, 1]
    corr_xz = correlacao_cruzada(dados)[0, 2]
    corr_yz = correlacao_cruzada(dados)[1, 2]
    # [amplitude_x, amplitude_y, amplitude_z], [variancia_x, variancia_y, variancia_z],
    # sem impacto [dist_euclidiana],[entropia_x, entropia_y, entropia_z],
    # Retorna todas as características concatenadas em um único vetor
    return np.concatenate([
        media, desvio, maximo, minimo, energia,  curtose,
        [acf_x, acf_y, acf_z], 
        [mav_x, mav_y, mav_z], [rms_x, rms_y, rms_z], 
        [poder_x, poder_y, poder_z], [corr_xy, corr_xz, corr_yz]
    ])

def tree(dados_treino,labels_treino,dados_teste,labels_teste):
    global modelo_rf
    X = []
    y = []

    # Extrair características das sequências temporais 3D
    for i in range(len(dados_treino)):
        X.append(extrair_caracteristicas(dados_treino[i]))  # Extrair características da sequência
        y.append(labels_treino[i])  # Etiqueta associada à sequência

    X = np.array(X)
    y = LabelEncoder().fit_transform(y)  # Codificar as labels para valores numéricos

    # 
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)


    X = []
    y = []
    for i in range(len(dados_teste)):
        X.append(extrair_caracteristicas(dados_teste[i]))  # Extrair características da sequência 
        y.append(labels_teste[i])  # Etiqueta associada à sequência

    y_pred = modelo_rf.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f'Precisao: {accuracy:.2f}')
    

    # Matriz de Confusão
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y, y_pred))
    
def modeloRandomTree(dados):
    dados_train, dados_test, dados_validation = divisaoDados(dados)

    # Preparar os dados de treino e teste
    X_train, Y_train = prepararDadosParaCNN(dados_train)
    X_test, Y_test = prepararDadosParaCNN(dados_test)

    # Passar os dados de treino para o modelo Random Forest
    tree(X_train, Y_train,X_test, Y_test)
    
def salvarModelo(str):
    global modelo
    global modelo_rf
    if str=="CNN":
        if modelo:
            # Garantir que a pasta 'modelos' exista, caso contrário, cria-a
            pasta_modelos = 'modelos'
            if not os.path.exists(pasta_modelos):
                os.makedirs(pasta_modelos)  # Cria a pasta 'modelos' se não existir

            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            nome = input("Introduza o nome do modelo: ")

            modelo.save(os.path.join(pasta_modelos, nome + '.keras'))  # Salva o modelo no formato H5 dentro da pasta 'modelos'
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Modelo salvo com sucesso em {os.path.join(pasta_modelos, nome + '.keras')}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print("Nenhum modelo CNN para salvar!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
    elif str=="Forest":
        if modelo_rf:
            # Garantir que a pasta 'modelos' exista, caso contrário, cria-a
            pasta_modelos = 'modelos'
            if not os.path.exists(pasta_modelos):
                os.makedirs(pasta_modelos)  # Cria a pasta 'modelos' se não existir

            
            nome = input("Introduza o nome do modelo: ")
            dump(modelo_rf, os.path.join(pasta_modelos, nome + '.joblib'))
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Modelo salvo com sucesso em {os.path.join(pasta_modelos, nome + '.joblib')}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print("Nenhum modelo Forest para salvar!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
    else:
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        print("Modelo nao reconhecido")
        print("-------------------------------------------------------------------------------------------------------------------------------------------")

def carregarModelo(str):
    if str=="CNN":
        # Caminho da pasta onde os modelos são armazenados
        pasta_modelos = 'modelos'
        
        # Verificar se a pasta existe
        if not os.path.exists(pasta_modelos):
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"A pasta '{pasta_modelos}' não existe!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return None

        # Solicitar o nome do modelo a ser carregado
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        nome = input("Introduza o nome do modelo a carregar (sem a extensão .keras): ")

        # Construir o caminho completo para o modelo
        caminho_modelo = os.path.join(pasta_modelos, nome + '.keras')

        # Verificar se o arquivo do modelo existe
        if os.path.exists(caminho_modelo):
            modelo_carregado = load_model(caminho_modelo)  # Carrega o modelo
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Modelo '{nome}' carregado com sucesso de {caminho_modelo}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return modelo_carregado
        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"O modelo '{nome}' não foi encontrado na pasta '{pasta_modelos}'!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return None
    elif str=="Forest":
        # Caminho da pasta onde os modelos são armazenados
        pasta_modelos = 'modelos'
        
        # Verificar se a pasta existe
        if not os.path.exists(pasta_modelos):
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"A pasta '{pasta_modelos}' não existe!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return None

        # Solicitar o nome do modelo a ser carregado
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        nome = input("Introduza o nome do modelo a carregar (sem a extensão .joblib): ")

        # Construir o caminho completo para o modelo
        caminho_modelo = os.path.join(pasta_modelos, nome + '.joblib')

        # Verificar se o arquivo do modelo existe
        if os.path.exists(caminho_modelo):
            modelo_carregado = load(caminho_modelo)     # Carrega o modelo
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Modelo '{nome}' carregado com sucesso de {caminho_modelo}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return modelo_carregado
        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"O modelo '{nome}' não foi encontrado na pasta '{pasta_modelos}'!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            return None
         
def testarModelo(str,dados):
    global modelo
    global modelo_rf
    global classes



    # Dividir os dados em treino, teste e validação
    dados_train, dados_test, dados_val = divisaoDados(dados)

    # Preparar os dados normalizados para ambos os modelos
    X_test, Y_test = prepararDadosParaCNN(dados_test)

    if str=="CNN":
        if modelo:
            cnn_loss, cnn_accuracy = modelo.evaluate(X_test, Y_test, verbose=0)
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Modelo CNN - Loss: {cnn_loss:.4f}, Accuracy: {cnn_accuracy:.4f}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")

            # Fazer previsões com o modelo CNN
            Y_pred_cnn = modelo.predict(X_test)
            Y_pred_classes_cnn = np.argmax(Y_pred_cnn, axis=1)

            # Matriz de confusão para CNN
            cm_cnn = confusion_matrix(Y_test, Y_pred_classes_cnn)

            plt.figure(figsize=(14, 6))
            # Matriz de confusão CNN
            plt.subplot(1, 1, 1)
            sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title('Matriz de Confusão - CNN')
            plt.xlabel('Classe Prevista')
            plt.ylabel('Classe Real')
            plt.tight_layout()
            plt.show()

        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print("Nenhum modelo CNN para usar!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
        
    elif str=="Forest":
        if modelo_rf:
             # Preparar as características para o Random Forest (usando o mesmo conjunto de dados normalizados)
            X_test_rf = [extrair_caracteristicas(x) for x in X_test]
            X_test_rf = np.array(X_test_rf)

            
            rf_accuracy = modelo_rf.score(X_test_rf, Y_test)
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Precisão (Accuracy): {rf_accuracy:.4f}")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")

            # Fazer previsões com o modelo Random Forest
            Y_pred_rf = modelo_rf.predict(X_test_rf)

            # Matriz de confusão para Random Forest
            cm_rf = confusion_matrix(Y_test, Y_pred_rf)

            # Plotar as matrizes de confusão
            plt.figure(figsize=(14, 6))

            # Matriz de confusão Random Forest
            plt.subplot(1, 1, 1)
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title('Matriz de Confusão - Random Forest')
            plt.xlabel('Classe Prevista')
            plt.ylabel('Classe Real')

            plt.tight_layout()
            plt.show()
            print(f"Modelo Random Forest - Accuracy: {rf_accuracy:.4f}")
        else:
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print("Nenhum modelo Forest para usar!")
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
       
def comparar_modelos(dados):
    global modelo
    global modelo_rf
    global classes

    if modelo is None or modelo_rf is None:
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        print("Certifique-se de que ambos os modelos (CNN e Random Forest) estão treinados antes de compará-los.")
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        return

    # Dividir os dados em treino, teste e validação
    dados_train, dados_test, dados_val = divisaoDados(dados)

    # Preparar os dados normalizados para ambos os modelos
    X_test, Y_test = prepararDadosParaCNN(dados_test)


    # Avaliar o modelo CNN
    print("\n== Avaliação do modelo CNN ==")
    cnn_loss, cnn_accuracy = modelo.evaluate(X_test, Y_test, verbose=0)
    print(f"Perda (Loss): {cnn_loss:.4f}, Precisão (Accuracy): {cnn_accuracy:.4f}")

    # Fazer previsões com o modelo CNN
    Y_pred_cnn = modelo.predict(X_test)
    Y_pred_classes_cnn = np.argmax(Y_pred_cnn, axis=1)

    # Matriz de confusão para CNN
    cm_cnn = confusion_matrix(Y_test, Y_pred_classes_cnn)

    # Preparar as características para o Random Forest (usando o mesmo conjunto de dados normalizados)
    X_test_rf = [extrair_caracteristicas(x) for x in X_test]
    X_test_rf = np.array(X_test_rf)

    # Avaliar o modelo Random Forest
    print("\n== Avaliação do modelo Random Forest ==")
    rf_accuracy = modelo_rf.score(X_test_rf, Y_test)
    print(f"Precisão (Accuracy): {rf_accuracy:.4f}")

    # Fazer previsões com o modelo Random Forest
    Y_pred_rf = modelo_rf.predict(X_test_rf)

    # Matriz de confusão para Random Forest
    cm_rf = confusion_matrix(Y_test, Y_pred_rf)

    # Plotar as matrizes de confusão
    plt.figure(figsize=(14, 6))

    # Matriz de confusão CNN
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão - CNN')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')

    # Matriz de confusão Random Forest
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão - Random Forest')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')

    plt.tight_layout()
    plt.show()

    # Comparação geral
    print("\n== Comparação Geral ==")
    print(f"Modelo CNN - Loss: {cnn_loss:.4f}, Accuracy: {cnn_accuracy:.4f}")
    print(f"Modelo Random Forest - Accuracy: {rf_accuracy:.4f}")

    print("\n** Comparação concluída. Consulte os gráficos para mais detalhes. **")
    print("-------------------------------------------------------------------------------------------------------------------------------------------")

def remover_atividade(atividade_remover, dados):
    global modelo
    global modelo_rf 
    # Verifica se a atividade existe nos dados
    atividades = [dado[0] for dado in dados]  # Considerando que a atividade está na primeira posição de cada linha

    if atividade_remover not in atividades:
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"A atividade '{atividade_remover}' não foi encontrada nos dados.")
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        return dados  # Retorna os dados originais se a atividade não existir

    # Filtra os dados para remover a atividade especificada
    dados_filtrados = [dado for dado in dados if dado[0] != atividade_remover]
    modelo =None
    modelo_rf =None
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"A atividade '{atividade_remover}' foi removida dos dados.")
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print("Modelos foram limpos!")
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
        
    return dados_filtrados 

def main():
    global count_error
    global min_amostras
    global janela_tempo
    global modelo
    global modelo_rf 
    global classes

    dados_originais=[]
    os.system("cls")
    dados_originais=leituraTodosDados(dados_originais)
    dados=dados_originais
    print(f"Numero de ficheiros rejeitados:{count_error}")
    while True:
        print("Menu:")
        print("Opçoes :")
        print("1-Modelo CNN")
        print("2-Guardar modelo CNN")
        print("3-Carregar modelo CNN")
        print("4-Usar modelo carregado CNN")
        print("5-Modelo Forest")
        print("6-Guardar modelo Forest")
        print("7-Carregar modelo Forest")
        print("8-Usar modelo carregado Forest")
        print("9-comparar modelos CNN vs Forest")
        print("r-remove uma das actividades")
        print("o-restaura os dados todos")
        print("p-print dos resultados dos modelos ate agora usados")
        print("m-mudar o numero minimo de amostras")
        print("j-mudar o numero janela de tempo a considerar")
        print("0-exit")
        print(classes)
        print(f"minimo de amostras ={min_amostras:4d}")
        print(f"janela de tempo={janela_tempo:4d}")
        c=input("Introduza a Opçao:")
        os.system("cls")
        if c=="1":
            modeloCNN(dados)
        elif c=="2":
            salvarModelo("CNN")         
        elif c=="3":
            modelo=carregarModelo("CNN")                 
        elif c=="4":
            testarModelo("CNN",dados)            
        elif c=="5":
            modeloRandomTree(dados)
        elif c=="6":
            salvarModelo("Forest")
        elif c=="7":
            modelo_rf=carregarModelo("Forest")
        elif c=="8":
            testarModelo("Forest",dados) 
        elif c=="9":
            comparar_modelos(dados) 
        elif c=="p":
            printDados(dados) 
        elif c=="r":
            actividade=input("actividade a remover:")
            dados=remover_atividade(actividade,dados)
        elif c=="o":    
            dados=dados_originais
            modelo =None
            modelo_rf =None
            print("-------------------------------------------------------------------------------------------------------------------------------------------")
            print("Dados Restaurados! Modelos foram limpos!") 
            print("-------------------------------------------------------------------------------------------------------------------------------------------") 
        elif c=="m":
            min_amostras=int(input("Novo minimo de amostras:"))
        elif c=="j":
            janela_tempo=int(input("Nova janela de tempo:"))
        elif c=="0":
            print("EXIT")
            break

main()
