def calculate_recall_precision(label, prediction):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
 
    for i in range(0, len(label)):
        if prediction[i] == 1:
            if prediction[i] == label[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction[i] == label[i]:
                true_negatives += 1
            else:
                false_negatives += 1
 
    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) \
               / (true_positives + true_negatives + false_positives + false_negatives)
 
    # precision is "how useful the search results are"
    precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the results are"
    recall = true_positives / (true_positives + false_negatives)
 
    f1_score = 2 / ((1 / precision) + (1 / recall))
 
    return accuracy, precision, recall, f1_score
 
# usage example:
y_true = [1, 1, 0, 1, 1,1,1,0]
y_pred = [0, 1, 0, 0, 1,0,0,0]
 
accuracy, precision, recall, f1_score = calculate_recall_precision(y_true, y_pred)
 

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
