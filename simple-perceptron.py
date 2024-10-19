import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

# Modificar a variável alvo para binária (0 para Setosa, 1 para todas as outras)
#  - Se for Iris Setosa, a classe será 0.
#  - Se for qualquer outra espécie será 1.
y = np.where(y == 0, 0, 1)

# Dividir os dados em conjunto de treino e teste
# O conjunto de treino será usado para ajustar os pesos da Perceptron
# O conjunto de teste será usado para avaliar o desempenho do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
# A padronização é importante para que todas as características tenham a mesma escala,
# o que facilita o aprendizado do modelo. Isso transforma os dados para que tenham média 0 e variância 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Inicializa os hiperparâmetros da Perceptron.
        
        Parâmetros:
        - learning_rate: Taxa de aprendizado (o quão grande será o ajuste dos pesos em cada iteração).
        - n_iters: Número de iterações que o algoritmo vai rodar sobre o conjunto de treino.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        # A função de ativação usada é a função de passo,
        # que retorna 0 ou 1.
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def _unit_step_function(self, x):
        """
        Função de ativação passo:
        Se a entrada for maior ou igual a 0, retorna 1; caso contrário, retorna 0.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Ajusta os pesos da Perceptron com base nos dados de treino.
        
        Parâmetros:
        - X: Conjunto de treinamento com as amostras (dados de entrada).
        - y: Labels (rótulos) correspondentes para cada amostra.
        """
        n_samples, n_features = X.shape

        # Inicializa os pesos e o bias com zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Laço de treinamento
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Combinação linear dos pesos e entradas mais o bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Aplicar a função de ativação para obter a predição
                y_pred = self.activation_func(linear_output)

                # Atualizamos os pesos e bias com base no erro (diferença entre y verdadeiro e y predito)
                update = self.lr * (y[idx] - y_pred)
                # O peso é ajustado proporcionalmente ao erro e à taxa de aprendizado
                self.weights += update * x_i
                # O bias também é ajustado
                self.bias += update

    def predict(self, X):
        """
        Faz previsões usando o modelo treinado.
        
        Parâmetros:
        - X: Conjunto de dados para os quais queremos prever as classes.
        
        Retorna:
        - y_pred: Predições binárias (0 ou 1) para cada amostra.
        """
        # Cálculo da saída linear e aplicação da função de ativação
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred
    '''
    Treinamento da Perceptron
    A taxa de aprendizado (learning_rate) controla o quanto ajustamos os pesos a cada passo
    n_iters define quantas vezes a Perceptron verá o conjunto de dados durante o treinamento
    '''
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)


# Fazer o teste do modelo
y_pred_test = perceptron.predict(X_test)

# Avaliar a performance da Perceptron
# Usarei apenas a métrica de acurácia (proporção de predições corretas)
test_acc = accuracy_score(y_test, y_pred_test)

# Exibir os resultados
print(f"Precisão no Teste: {test_acc * 100:.2f}%")