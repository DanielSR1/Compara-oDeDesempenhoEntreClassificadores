from dataset import*
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
gnb=GaussianNB
gnb.fit(X_train, y_train)
esperado=y_test
predito=gnb.predict(X_test)
print('relatório: %s\n%s\n' % (gnb, metrics.classification_report(esperado, predito)))