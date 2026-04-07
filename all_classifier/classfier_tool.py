import warnings

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def test_data(X, y, *,test_size: float = .2, random_state: int = 42, scale_data: bool = True) -> pd.DataFrame:
    """
    Test multiple classification models and return their performance metric values.
    :param X: Feature matrix
    :param y: Target vector
    :param test_size: Proportion of the dataset to include in the test set
    :param random_state: Random seed
    :param scale_data: Whether to scale the data before fitting models.
    :return:
    DataFrame: Result comparing all classifiers
    """

    # Veriyi eğitim ve test setlerine böl (stratified split ile sınıf dengesini koru)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    # Eğer ölçeklendirme istenirse, StandardScaler ile verileri normalize et
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    # Kullanılacak 14 farklı sınıflandırma modelini listele
    models = [
        ('Logistic Regression', LogisticRegression(random_state=random_state, max_iter=len(y))),
        ('Decision Tree', DecisionTreeClassifier(random_state=random_state)),
        ('Random Forest', RandomForestClassifier(random_state=random_state)),
        ('SVM', SVC(random_state=random_state)),
        ('KNN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=random_state)),
        ('AdaBoost', AdaBoostClassifier(random_state=random_state)),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
        ('MLP Neural Network', MLPClassifier(random_state=random_state, max_iter=len(y))),
        ('SGD Classifier', SGDClassifier(random_state=random_state)),
        ('XGBoost Classifier', XGBClassifier(random_state=random_state)),
        ('LightGBM Classifier', LGBMClassifier(random_state=random_state,verbosity=-1))
    ]

    # Sonuçları depolamak için boş bir liste oluştur
    results = []

    # Her model için döngü başlat
    for name, model in models:
        try:
            # Modeli eğitim verisiyle eğit
            model.fit(X_train, y_train)
            # Test verisi üzerinde tahmin yap
            y_pred = model.predict(X_test)
            # Doğruluk (accuracy) metriğini hesapla
            acc = accuracy_score(y_test, y_pred)
            # Precision, Recall, F1-score gibi detaylı metrikleri al
            cr = classification_report(y_test, y_pred, output_dict=True)

            # Modelin performans metriklerini sonuçlar listesine ekle
            results.append({
                'Classifier': name,
                'Accuracy': acc,
                'Precision': cr['weighted avg']['precision'],
                'Recall': cr['weighted avg']['recall'],
                'F1-score': cr['weighted avg']['f1-score']
            })
        except Exception as ex:
            # Model eğitiminde hata oluşursa, None değerleriyle sonuç ekle
            results.append({
                'Classifier': name,
                'Accuracy': acc,
                'Precision': None,
                'Recall': None,
                'F1-score': None
            })

    # Sonuçları pandas DataFrame'e dönüştür
    results_df = pd.DataFrame(results)
    # DataFrame'i doğruluk değerine göre büyükten küçüğe sırala
    results_df = results_df.sort_values('Accuracy', ascending=False)
    # Sıralanmış sonuçları döndür
    return results_df

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target
    response = test_data(data, target,test_size=.3)
    print(response)