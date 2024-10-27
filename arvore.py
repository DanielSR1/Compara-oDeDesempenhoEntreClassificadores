from dataset import*
from sklearn import tree
modelo=tree.DecisionTreeClassifier()
modelo.fit(X_train, y_train)
esperado=X_test
predito=modelo.predict(X_test)
print('relatório: %s\n%s\n' % (modelo, metrics.classification_report(esperado, predito)))