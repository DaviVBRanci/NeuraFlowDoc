import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

class RedeNeural:
    def __init__(self, camadas=[5], max_iter=1000, solver='adam', regularizacao='l2', C=1.0):
        self.camadas = camadas
        self.max_iter = max_iter
        self.solver = solver
        self.regularizacao = regularizacao
        self.C = C
        self.model = None
        self.historia = []

    def treinar(self, X, y):
        if self.regularizacao == 'l2':
            self.model = MLPClassifier(hidden_layer_sizes=self.camadas, max_iter=self.max_iter, solver=self.solver, alpha=self.C)
        elif self.regularizacao == 'l1':
            self.model = LogisticRegression(penalty='l1', C=self.C)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=self.camadas, max_iter=self.max_iter, solver=self.solver)
        self.model.fit(X, y)
        self.historia = self.model.loss_curve_

    def testar(self, X, y):
        if self.model is None:
            print("Modelo não treinado.")
            return
        y_pred = self.model.predict(X)
        acuracia = accuracy_score(y, y_pred)
        print(f'Acurácia: {acuracia * 100:.2f}%')

    def visualizar_aprendizado(self):
        if not self.historia:
            print("Histórico vazio!")
            return
        # Ajustar a visualização para garantir que o gráfico seja gerado mesmo se houver discrepâncias de tamanho
        iteracoes = min(self.max_iter, len(self.historia))
        plt.plot(range(iteracoes), self.historia[:iteracoes])
        plt.title("Curva de Aprendizado")
        plt.xlabel("Épocas")
        plt.ylabel("Perda")
        plt.show()

    def analise_importancia(self, X):
        if hasattr(self.model, 'coef_'):
            importancias = self.model.coef_[0]
            plt.bar(range(len(importancias)), importancias)
            plt.title("Importância das Características")
            plt.xlabel("Características")
            plt.ylabel("Importância")
            plt.show()
        else:
            print("Modelo sem coeficientes.")

    def salvar_modelo(self, nome_arquivo="modelo_treinado.pkl"):
        joblib.dump(self.model, nome_arquivo)

    def carregar_modelo(self, nome_arquivo="modelo_treinado.pkl"):
        self.model = joblib.load(nome_arquivo)

    def analise_desempenho(self, X, y):
        y_pred = self.model.predict(X)
        print("Relatório de Desempenho:")
        print(classification_report(y, y_pred))

    def avaliacao_regressao(self, X, y):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"Erro Quadrático Médio: {mse}")
        print(f"R²: {r2}")

    def cross_validation(self, X, y, k=5):
        scores = cross_val_score(self.model, X, y, cv=k)
        print(f'Validação Cruzada - Acurácia média: {scores.mean() * 100:.2f}%')

    def ajuste_hiperparametros(self, X, y):
        param_grid = {'hidden_layer_sizes': [(10, 5), (20, 10)], 'max_iter': [500, 1000], 'solver': ['adam', 'sgd']}
        grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=3)
        grid_search.fit(X, y)
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def gerar_dados_aleatorios(self, n_amostras=1000, n_features=20, n_classes=2):
        X = np.random.rand(n_amostras, n_features)
        y = np.random.randint(0, n_classes, size=n_amostras)
        return X, y

    def normalizar_dados(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def aprender_completo(self, X, y):
        self.treinar(X, y)
        self.testar(X, y)
        self.visualizar_aprendizado()
        self.analise_importancia(X)
        self.analise_desempenho(X, y)

    def gerar_grafico_curva_aprendizado(self):
        plt.plot(range(self.max_iter), self.historia)
        plt.title("Curva de Aprendizado")
        plt.xlabel("Épocas")
        plt.ylabel("Perda")
        plt.show()

    def aprender_semi_supervisionado(self, X_rotulados, y_rotulados, X_nao_rotulados):
        modelo = LabelSpreading(kernel='knn')
        modelo.fit(X_rotulados, y_rotulados)
        y_pred = modelo.predict(X_nao_rotulados)
        return y_pred

    def limpar_dados(self, X):
        return np.nan_to_num(X)

    def balancear_classes(self, X, y):
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def calcular_importancia_gini(self, X, y):
        model = DecisionTreeClassifier()
        model.fit(X, y)
        importancias = model.feature_importances_
        plt.bar(range(len(importancias)), importancias)
        plt.title("Importância das Características (Gini)")
        plt.xlabel("Características")
        plt.ylabel("Importância")
        plt.show()

    def otimizar_modelo(self, X, y):
        param_grid = {'hidden_layer_sizes': [(10, 5), (20, 10)], 'max_iter': [500, 1000], 'solver': ['adam', 'sgd']}
        grid_search = GridSearchCV(self.model, param_grid, cv=3)
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def ajustar_taxa_aprendizado(self, X, y, taxa=0.001):
        self.model.set_params(learning_rate_init=taxa)
        self.treinar(X, y)

    def reduzir_dimensionais(self, X, n_componentes=2):
        pca = PCA(n_components=n_componentes)
        return pca.fit_transform(X)

    def prever_probalidades(self, X):
        return self.model.predict_proba(X)

    def transformar_dados(self, X):
        return self.model.transform(X)

    def atualizar_modelo(self, X, y):
        self.model.partial_fit(X, y)

    def calcular_erro(self, X, y):
        return mean_squared_error(y, self.model.predict(X))

    def adicionar_funcoes(self, X):
        return np.concatenate([X, X ** 2], axis=1)

    def validar_modelo(self, X, y):
        return accuracy_score(y, self.model.predict(X))

    def verificar_dados_faltantes(self, X):
        return np.any(np.isnan(X))

    def ajustar_parametro(self, parametro, valor):
        self.model.set_params(**{parametro: valor})

    def ajustar_regularizacao(self, regularizacao='l1'):
        self.regularizacao = regularizacao
        if self.regularizacao == 'l2':
            self.model = MLPClassifier(hidden_layer_sizes=self.camadas, max_iter=self.max_iter, solver=self.solver, alpha=self.C)
        elif self.regularizacao == 'l1':
            self.model = LogisticRegression(penalty='l1', C=self.C)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=self.camadas, max_iter=self.max_iter, solver=self.solver)

    def exibir_modelo(self):
        print(self.model)

    def gerar_dados_classificados(self, n_amostras=1000, n_classes=2):
        X, y = self.gerar_dados_aleatorios(n_amostras, 20, n_classes)
        return X, y

    def treinar_em_batch(self, X, y):
        X, y = shuffle(X, y)
        self.treinar(X, y)

    def calcular_confusao(self, X, y):
        from sklearn.metrics import confusion_matrix
        y_pred = self.model.predict(X)
        return confusion_matrix(y, y_pred)

    def obter_acuracia_teste(self, X, y):
        return accuracy_score(y, self.model.predict(X))

    def obter_acuracia_valida(self, X, y):
        return cross_val_score(self.model, X, y, cv=5).mean()

    def otimizar_taxa_aprendizado_e_max_iter(self, X, y):
        param_grid = {'max_iter': [500, 1000], 'learning_rate_init': [0.001, 0.01, 0.1]}
        grid_search = GridSearchCV(self.model, param_grid, cv=3)
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def treinar_todos_dados(self, X, y):
        self.treinar(X, y)

    def salvar_pesos(self, nome_arquivo="pesos_modelo.pkl"):
        joblib.dump(self.model.coefs_, nome_arquivo)

    def carregar_pesos(self, nome_arquivo="pesos_modelo.pkl"):
        self.model.coefs_ = joblib.load(nome_arquivo)

    def gerenciar_dados_faltantes(self, X):
        return np.nan_to_num(X)

    def visualizar_importancia_features(self, X):
        self.analise_importancia(X)

    def realizar_previsao(self, X):
        return self.model.predict(X)
    def preprocessar_dados(self, X):
        # Aplicação de técnicas de pré-processamento, como normalização e limpeza
        X_limpo = self.limpar_dados(X)
        X_normalizado = self.normalizar_dados(X_limpo)
        return X_normalizado

    def criar_matriz_confusao(self, X, y):
        from sklearn.metrics import confusion_matrix
        y_pred = self.model.predict(X)
        return confusion_matrix(y, y_pred)

    def balancear_classes_smote(self, X, y):
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def realizar_classificacao(self, X, y):
        self.treinar(X, y)
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred)

    def visualizar_pca(self, X):
        pca = PCA(n_components=2)
        X_reduzido = pca.fit_transform(X)
        plt.scatter(X_reduzido[:, 0], X_reduzido[:, 1])
        plt.title('Redução de Dimensionalidade com PCA')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.show()

    def realizar_ajuste(self, X, y):
        self.ajuste_hiperparametros(X, y)

    def aplicar_regularizacao(self, X, y):
        self.ajustar_regularizacao(self.regularizacao)
        self.treinar(X, y)

    def estimar_dados_faltantes(self, X):
        # Exemplo de como utilizar uma técnica para preencher dados faltantes
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        return imputer.fit_transform(X)

    def ajustar_novos_parametros(self, X, y, novos_parametros):
        self.model.set_params(**novos_parametros)
        self.treinar(X, y)

    def aplicar_dropout(self, X):
        # Exemplo de simulação de dropout para redes neurais
        dropout = np.random.rand(*X.shape) < 0.5
        return X * dropout

    def aplicar_early_stopping(self, X, y):
        # Exemplo simples de Early Stopping, para parar o treinamento se o erro não melhorar
        min_loss = np.inf
        for i in range(self.max_iter):
            self.treinar(X, y)
            loss = self.model.loss_curve_[-1]
            if loss < min_loss:
                min_loss = loss
            else:
                break

    def gerar_rna(self, X, y):
        # Função para gerar a rede neural com os parâmetros de configuração
        self.treinar(X, y)
        return self.model

    def realizar_ajuste_hiperparametros_completo(self, X, y):
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (30,)],
            'solver': ['adam', 'sgd'],
            'max_iter': [500, 1000],
            'learning_rate_init': [0.001, 0.01]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def fazer_previsao_com_proba(self, X):
        return self.model.predict_proba(X)

    def treinar_com_batch(self, X, y):
        X, y = shuffle(X, y)
        self.treinar(X, y)

    def obter_erro_quadratico(self, X, y):
        return mean_squared_error(y, self.model.predict(X))

    def ajustar_com_loss(self, X, y):
        # Ajuste com base na perda
        self.model.set_params(learning_rate_init=0.01)
        self.treinar(X, y)

    def visualizar_curva_perda(self):
        # Exemplo para visualização da curva de perda durante o treinamento
        plt.plot(self.historia)
        plt.title('Curva de Perda durante o Treinamento')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.show()

    def verificar_regularizacao(self):
        if self.regularizacao:
            print(f"Regularização: {self.regularizacao}")
        else:
            print("Sem regularização.")

    def realizar_predicao_classificao(self, X):
        return self.model.predict(X)

    def aplicar_deep_learning(self, X, y):
        # Exemplo de uma rede neural profunda
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, solver='adam')
        model.fit(X, y)
        return model

    def salvar_modelo_em_treinamento(self, arquivo="modelo_treinado.pkl"):
        joblib.dump(self.model, arquivo)

    def carregar_modelo_em_treinamento(self, arquivo="modelo_treinado.pkl"):
        self.model = joblib.load(arquivo)

    def classificar_dados(self, X, y):
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred)

    def calcular_acuracia(self, X, y):
        return accuracy_score(y, self.model.predict(X))

    def avaliar_modelo(self, X, y):
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred)

    def plotar_dados(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Distribuição dos Dados")
        plt.show()

    def ajustar_taxa_aprendizado_otimizada(self, X, y):
        param_grid = {'learning_rate_init': [0.01, 0.001]}
        grid_search = GridSearchCV(self.model, param_grid, cv=3)
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def implementar_aprendizado_semi_supervisionado(self, X_rotulados, y_rotulados, X_nao_rotulados):
        modelo = LabelSpreading(kernel='knn')
        modelo.fit(X_rotulados, y_rotulados)
        return modelo.predict(X_nao_rotulados)

    def atualizar_metrica(self, X, y):
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)

    def ajustar_inclusao_features(self, X, nova_feature):
        return np.concatenate([X, nova_feature], axis=1)

    def estimar_novos_dados(self, X):
        return self.model.predict(X)
    def realizar_ajuste_treinamento(self, X, y, n_iter=1000, learning_rate=0.001):
        # Ajuste do treinamento com base no número de iterações e taxa de aprendizado
        self.model.set_params(max_iter=n_iter, learning_rate_init=learning_rate)
        self.treinar(X, y)

    def realizar_aprendizado_online(self, X, y):
        # Treinamento online com mini-batches
        for i in range(0, len(X), self.batch_size):
            X_batch, y_batch = X[i:i+self.batch_size], y[i:i+self.batch_size]
            self.treinar(X_batch, y_batch)

    def gerar_mapa_calor(self, X, y):
        # Gerar um mapa de calor para visualização das correlações entre as features
        import seaborn as sns
        import pandas as pd
        corr = pd.DataFrame(X).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Mapa de Calor das Correlações')
        plt.show()

    def calcular_erro_classificacao(self, X, y):
        y_pred = self.model.predict(X)
        return 1 - accuracy_score(y, y_pred)

    def treinar_com_regulacao_l2(self, X, y):
        # Aplicando regularização L2 para evitar overfitting
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2', solver='liblinear')
        model.fit(X, y)
        return model

    def aplicar_extracao_features(self, X):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(X)

    def realizar_classificacao_multiclasse(self, X, y):
        from sklearn.multioutput import MultiOutputClassifier
        model = MultiOutputClassifier(self.model)
        model.fit(X, y)
        return model.predict(X)

    def implementar_redundancia_dados(self, X):
        return np.concatenate([X, X], axis=0)

    def realizar_ajuste_simples(self, X, y, parametros={}):
        # Ajuste simples de parâmetros
        self.model.set_params(**parametros)
        self.treinar(X, y)

    def realizar_ajuste_com_modelo_final(self, X, y):
        self.treinar(X, y)
        return self.model

    def gerar_grafico_barras(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        counts = np.bincount(y_encoded)
        plt.bar(encoder.classes_, counts)
        plt.title('Distribuição de Classes')
        plt.xlabel('Classe')
        plt.ylabel('Quantidade')
        plt.show()

    def realizar_ajuste_hiperparametros_com_random_search(self, X, y):
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'max_iter': [100, 500, 1000]}
        random_search = RandomizedSearchCV(self.model, param_dist, cv=5)
        random_search.fit(X, y)
        return random_search.best_params_

    def treinar_com_cross_validation(self, X, y, n_folds=5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, X, y, cv=n_folds)
        return scores.mean()

    def calcular_erro_quadratico_medio(self, X, y):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y, self.model.predict(X))

    def ajustar_aprendizado_online(self, X, y, n_iter=1000):
        # Ajuste do modelo com aprendizado online
        for i in range(n_iter):
            X_batch, y_batch = shuffle(X, y)
            self.treinar(X_batch, y_batch)

    def avaliar_com_metricas(self, X, y):
        from sklearn.metrics import classification_report
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred)

    def realizar_classificacao_com_xgboost(self, X, y):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.fit(X, y)
        return model.predict(X)

    def treinar_com_early_stopping(self, X, y, n_iter=1000, patience=10):
        # Implementação de early stopping para interromper o treinamento antes de overfitting
        prev_loss = np.inf
        for i in range(n_iter):
            self.treinar(X, y)
            loss = self.model.loss_curve_[-1]
            if loss > prev_loss:
                patience -= 1
                if patience == 0:
                    break
            prev_loss = loss

    def realizar_ajuste_com_k_fold(self, X, y, n_splits=5):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_splits)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.treinar(X_train, y_train)
            score = self.model.score(X_test, y_test)
        return score

    def gerar_gerenciamento_com_plot(self, X, y):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()

    def realizar_ajuste_learning_rate(self, X, y, taxa_aprendizado):
        self.model.set_params(learning_rate_init=taxa_aprendizado)
        self.treinar(X, y)
    def gerar_grafico_aprendizado(self, X, y):
        # Gerar um gráfico mostrando o progresso do aprendizado
        loss_curve = self.model.loss_curve_
        plt.plot(loss_curve)
        plt.title('Progresso do Aprendizado')
        plt.xlabel('Iterações')
        plt.ylabel('Perda')
        plt.show()

    def realizar_classificacao_binaria(self, X, y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X, y)
        return model.predict(X)

    def realizar_aprendizado_pareado(self, X, y, modelo_pareado):
        # Treinamento utilizando aprendizado pareado
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(X, y)
        return knn.predict(X)

    def calcular_erro_mean_absolute(self, X, y):
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y, self.model.predict(X))

    def calcular_indice_de_silhueta(self, X, y):
        from sklearn.metrics import silhouette_score
        return silhouette_score(X, y)

    def realizar_ajuste_com_grid_search(self, X, y):
        from sklearn.model_selection import GridSearchCV
        param_grid = {'hidden_layer_sizes': [(10,), (50,)], 'activation': ['relu', 'tanh']}
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def treinar_com_batch_normalization(self, X, y):
        # Normalização em lote durante o treinamento
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.treinar(X_scaled, y)

    def realizar_classificacao_com_svm(self, X, y):
        from sklearn.svm import SVC
        model = SVC()
        model.fit(X, y)
        return model.predict(X)

    def realizar_ajuste_treinado_multimodal(self, X1, X2, y):
        # Treinamento utilizando múltiplas fontes de dados (dados multimodais)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        X = np.concatenate([X1, X2], axis=1)
        model.fit(X, y)
        return model.predict(X)

    def calcular_matriz_covariancia(self, X):
        # Calcular a matriz de covariância
        return np.cov(X.T)

    def realizar_treinamento_com_early_stopping_e_monitoramento(self, X, y, n_iter=1000, patience=10):
        # Early stopping com monitoramento de métricas durante o treinamento
        prev_loss = np.inf
        for i in range(n_iter):
            self.treinar(X, y)
            loss = self.model.loss_curve_[-1]
            if loss > prev_loss:
                patience -= 1
                if patience == 0:
                    break
            prev_loss = loss

    def realizar_treinamento_com_regularizacao_lasso(self, X, y):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
        model.fit(X, y)
        return model.predict(X)

    def realizar_classificacao_com_decision_tree(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model.predict(X)

    def realizar_ajuste_com_podamento(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(ccp_alpha=0.01)
        model.fit(X, y)
        return model

    def gerar_grafico_curva_roc(self, X, y):
        from sklearn.metrics import roc_curve, auc
        y_pred = self.model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.show()

    def realizar_aprendizado_com_matriz_confusao(self, X, y):
        from sklearn.metrics import confusion_matrix
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        return cm

    def realizar_ajuste_treinamento_simplificado(self, X, y, n_iter=500):
        self.model.set_params(max_iter=n_iter)
        self.treinar(X, y)

    def realizar_ajuste_treinamento_em_modo_reduzido(self, X, y, n_iter=100):
        self.model.set_params(max_iter=n_iter, tol=1e-4)
        self.treinar(X, y)

    def realizar_classificacao_com_knn(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(X, y)
        return model.predict(X)

    def realizar_aprendizado_com_dados_faltantes(self, X, y):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        self.treinar(X_imputed, y)
def realizar_ajuste_treinamento_em_modo_reduzido(X, y, model, n_iter=100):
    """
    Ajusta o modelo para treinamento com um número reduzido de iterações.
    """
    model.set_params(max_iter=n_iter, tol=1e-4)
    model.fit(X, y)

def realizar_classificacao_com_knn(X, y):
    """
    Realiza a classificação com o algoritmo K-Nearest Neighbors (KNN).
    """
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X, y)
    return model.predict(X)

def realizar_aprendizado_com_dados_faltantes(X, y, model):
    """
    Realiza o aprendizado em dados com valores faltantes, substituindo-os pela média.
    """
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    model.fit(X_imputed, y)

def realizar_ajuste_com_svm(X, y):
    """
    Realiza o ajuste de um modelo de classificação usando Support Vector Machine (SVM).
    """
    from sklearn.svm import SVC
    model = SVC()
    model.fit(X, y)
    return model.predict(X)

def realizar_classificacao_com_random_forest(X, y):
    """
    Realiza a classificação usando o modelo Random Forest.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    return model.predict(X)

def realizar_ajuste_com_gradient_boosting(X, y):
    """
    Realiza o ajuste usando o modelo Gradient Boosting.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model.predict(X)

def realizar_classificacao_com_naive_bayes(X, y):
    """
    Realiza a classificação usando o algoritmo Naive Bayes.
    """
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X, y)
    return model.predict(X)

def realizar_classificacao_com_decision_tree(X, y):
    """
    Realiza a classificação usando o modelo Decision Tree.
    """
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model.predict(X)

def realizar_ajuste_com_logistic_regression(X, y):
    """
    Realiza o ajuste usando o modelo de Regressão Logística.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict(X)

def realizar_treinamento_com_cross_validation(X, y, model, cv=5):
    """
    Realiza o treinamento utilizando validação cruzada.
    """
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()

def gerar_grafico_distribuicao(X):
    """
    Gera um gráfico de distribuição dos dados fornecidos.
    """
    import matplotlib.pyplot as plt
    plt.hist(X, bins=50)
    plt.title("Distribuição dos Dados")
    plt.xlabel("Valor")
    plt.ylabel("Frequência")
    plt.show()

def realizar_classificacao_com_svm(X, y):
    """
    Realiza a classificação utilizando o modelo Support Vector Machine (SVM).
    """
    from sklearn.svm import SVC
    model = SVC()
    model.fit(X, y)
    return model.predict(X)

def calcular_erro_quadratico_medio(X, y, model):
    """
    Calcula o erro quadrático médio (MSE) do modelo.
    """
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y, model.predict(X))

def calcular_erro_absolute(X, y, model):
    """
    Calcula o erro absoluto médio (MAE) do modelo.
    """
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y, model.predict(X))

def realizar_ajuste_com_ridge_regression(X, y):
    """
    Realiza o ajuste utilizando a técnica de regressão Ridge.
    """
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1)
    model.fit(X, y)
    return model.predict(X)

def realizar_aprendizado_com_ensemble(X, y):
    """
    Realiza aprendizado com ensemble utilizando vários classificadores.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    clf1 = LogisticRegression()
    clf2 = SVC()
    ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2)], voting='hard')
    ensemble_model.fit(X, y)
    return ensemble_model.predict(X)

def gerar_curva_aprendizado(X, y, model):
    """
    Gera a curva de aprendizado para avaliar a performance do modelo.
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Treinamento")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validação")
    plt.title('Curva de Aprendizado')
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()
def listar_funcoes():
    """
    Lista todas as funções do código com suas descrições.
    """
    import inspect
    functions = inspect.getmembers(inspect.currentframe().f_globals, predicate=inspect.isfunction)
    func_list = []
    
    for func_name, func_obj in functions:
        doc_string = inspect.getdoc(func_obj) or "Sem descrição disponível"
        func_list.append((func_name, doc_string))
    
    for func_name, doc_string in func_list:
        print(f"Função: {func_name}")
        print(f"Descrição: {doc_string}")
        print("-" * 50)

# Exemplo de uso:
listar_funcoes()

def preparar_dados(X, Y):
    """
    Converte os dados para o formato correto para treinamento de redes neurais.
    
    Parâmetros:
    X : lista ou array
        Dados de entrada para a rede neural.
    Y : lista ou array
        Dados de saída (labels) para a rede neural.
    
    Retorna:
    X_2d : lista
        Dados de entrada em formato 2D.
    Y_1d : lista
        Dados de saída em formato 1D.
    """
    # Convertendo X para o formato 2D
    X_2d = [[x] for x in X]
    
    # Convertendo Y para o formato 1D
    Y_1d = [y[0] for y in Y] if isinstance(Y[0], list) else Y
    
    return X_2d, Y_1d
import matplotlib.pyplot as plt  # Importando dentro da função, sem precisar importar no código do usuário

def preparar_dados_e_plotar(X, Y, rede):
    """
    Converte os dados para o formato correto para treinamento de redes neurais e gera um gráfico do erro.
    
    Parâmetros:
    X : lista ou array
        Dados de entrada para a rede neural.
    Y : lista ou array
        Dados de saída (labels) para a rede neural.
    rede : objeto da rede neural
        A instância da rede neural que será treinada.
    
    Retorna:
    X_2d : lista
        Dados de entrada em formato 2D.
    Y_1d : lista
        Dados de saída em formato 1D.
    """
    # Convertendo X para o formato 2D (caso seja uma lista simples, transforma cada valor em uma lista)
    X_2d = [[x] if isinstance(x, (int, float)) else x for x in X]
    
    # Convertendo Y para o formato 1D, se Y for uma lista de listas, achata para 1D
    Y_1d = [y[0] if isinstance(y, list) else y for y in Y]
    
    # Variáveis para armazenar o progresso
    iteracoes = []
    erros = []

    # Função de callback para registrar o erro a cada iteração
    def registrar_erro(iteracao, erro):
        iteracoes.append(iteracao)
        erros.append(erro)

    # Treinando a rede e registrando o erro
    rede.treinar(X_2d, Y_1d, callback=registrar_erro)

    # Gerando o gráfico do erro ao longo das iterações
    plt.plot(iteracoes, erros, label="Erro de Treinamento", color='b')  # Usando uma linha azul
    plt.xlabel("Iterações")
    plt.ylabel("Erro")
    plt.title("Erro de Treinamento ao Longo das Iterações")
    plt.legend()
    plt.grid(True)
    plt.show()

    return X_2d, Y_1d

class RedeNeural:
    # Supondo que a rede já tenha sido definida com as camadas e pesos, etc.

    def __init__(self, camadas, max_iter):
        self.camadas = camadas
        self.max_iter = max_iter
        # Aqui você inicializaria seus pesos e outras variáveis necessárias

    def treinar(self, X, Y):
        # Função de treinamento (já implementada por você)
        pass

    def prever(self, X):
        """
        Método para fazer previsões com a rede neural.
        X : entrada para a rede neural.
        
        Retorna a previsão feita pela rede.
        """
        # A implementação do método 'prever' dependeria da arquitetura e dos pesos da sua rede neural.
        # Aqui, uma implementação simples de feedforward poderia ser assim:
        
        for camada in self.camadas:
            X = np.dot(X, camada.pesos)  # Supondo que 'camada.pesos' seja a matriz de pesos de cada camada
            X = self.função_ativação(X)  # Aplique uma função de ativação (ex: ReLU, Sigmoid)

        return X

    def função_ativação(self, X):
        # Exemplo de função de ativação (ReLU ou Sigmoid)
        return np.maximum(0, X)  # ReLU, por exemplo
