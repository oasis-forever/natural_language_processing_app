from sklearn.metrics import classification_report

def report(y_true, y_pred):
    return classification_report(y_true, y_pred)
