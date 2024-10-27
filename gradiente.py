from dataset import*
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
modelo=SGDClassifier
modelo.fit(X_train, y_train)
esperado=y_test
predito=modelo.predict(X_test)
print('relatório: %s\n%s\n' % (modelo, metrics.classification_report(esperado, predito)))


