import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def main():

# 1. Carrega e explora o dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Primeiras 5 linhas do dataset Iris:")
    print(df.head(), "\n")

# 2. Divide em treino e teste
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

# 3. Ajustar o modelo KNN com K=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

# 4. Fazer previsões e avaliar
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_report = metrics.classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )

    print(f"Acurácia: {accuracy:.2f}")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print("\nRelatório de Classificação:")
    print(class_report)

# 5. Visualização: dispersão das pétalas (verdadeiro vs previsto)
    plt.figure()
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, edgecolor='k')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.title('True Species')
    plt.show()

    plt.figure()
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, edgecolor='k')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.title('Predicted Species')
    plt.show()

if __name__ == '__main__':
    main()