from sklearn.metrics import precision_score, recall_score

def eval_each_class_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)

def eval_average_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro")

def eval_each_class_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)

def eval_average_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")
