import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Carregar datasets diretamente dos arquivos CSV
red_wine_data = pd.read_csv("C:/Users/alexr/Documents/wine+quality/winequality-red.csv", sep=';')
white_wine_data = pd.read_csv("C:/Users/alexr/Documents/wine+quality/winequality-white.csv", sep=';')

# Adicionar a coluna 'color' para diferenciar
red_wine_data['color'] = 1   
white_wine_data['color'] = 0   

# Combinar os dois datasets
wine_data = pd.concat([red_wine_data, white_wine_data], axis=0)

# Separar a variável alvo (y) e os atributos (X)
X = wine_data.drop(columns=['color'])   
y = wine_data['color'].to_numpy()  # Converter y para um array NumPy

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir a classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = 0

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)
 
                update = self.lr * (y[idx] - y_pred)  
                self.weights += update * x_i 
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

# Treinar a Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# Fazer previsões
y_pred_test = perceptron.predict(X_test)

# Avaliar a acurácia
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)

print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show() 

# Visualizar a Superfície de Decisão 
X_train_subset = X_train[:, :12]   
x_min, x_max = X_train_subset[:, 0].min() - 1, X_train_subset[:, 0].max() + 1
y_min, y_max = X_train_subset[:, 1].min() - 1, X_train_subset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))

# Transformando xx e yy para usar com o modelo
Z = perceptron.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel(), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()), 
                                               np.zeros_like(xx.ravel()),
                                               np.zeros_like(xx.ravel()),
                                               np.zeros_like(xx.ravel()),
                                               np.zeros_like(xx.ravel())]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_subset[:, 0], X_train_subset[:, 1], c=y_train, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Superfície de Decisão do Perceptron')
plt.show()