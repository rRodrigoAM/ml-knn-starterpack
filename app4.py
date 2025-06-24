import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def main():
    # 1) Dados
    np.random.seed(42)
    X = np.random.uniform(50, 200, 50).reshape(-1, 1)  # metragem (m²)
    y = 1500 + 80 * X.ravel() + np.random.normal(0, 5000, 50)  # preço R$

    # 2) Split treino/teste
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3) Treina modelo
    model = LinearRegression().fit(X_tr, y_tr)

    # 4) Avalia
    y_pred = model.predict(X_te)
    print(f"Precisão Geral = {r2_score(y_te, y_pred):.2f}")
    print(f"Erro ao quadrado = {mean_squared_error(y_te, y_pred):.2f}")

    # 5) Gráfico dispersão + linha de regressão
    plt.scatter(X_te, y_te, label="Dados reais")
    plt.plot(X_te, y_pred, color='r', label="Reta predita")
    plt.xlabel("Metragem (m²)")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.title("Regressão Linear Simples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()