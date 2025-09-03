import csv
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    data = pd.read_csv(filename)
    month_map = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'June':5, 'Jul':6, 
                 'Aug':7, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}
    data['Month'] = data['Month'].map(month_map)
    
    data['VisitorType'] = data['VisitorType'].apply(
        lambda x: 1 if x == 'Returning_Visitor' else 0
    )
    
    data['Weekend'] = data['Weekend'].apply(
        lambda x: 1 if x == 'TRUE' else 0
    )
    
    data['Revenue'] = data['Revenue'].apply(
        lambda x: 1 if x == 'TRUE' else 0
    )

     # 确保所有数值列都是正确的类型
    numeric_columns = [
        'Administrative', 'Administrative_Duration', 'Informational', 
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
    ]
    
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 处理可能的NaN值（如果有）
    data = data.dropna()
    
    evidence=data.drop(['Revenue'],axis=1).values.tolist()
    labels=data['Revenue'].values.tolist()
    return(evidence,labels)
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

def train_model(evidence, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels) 
    return neigh
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    

def evaluate(labels, predictions):
# 转换为 numpy 数组以便于计算
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # 计算真正例、真反例、假正例、假反例
    true_positives = np.sum((labels == 1) & (predictions == 1))
    true_negatives = np.sum((labels == 0) & (predictions == 0))
    false_positives = np.sum((labels == 0) & (predictions == 1))
    false_negatives = np.sum((labels == 1) & (predictions == 0))
    
    # 计算敏感性和特异性
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return sensitivity, specificity
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

if  __name__ == "__main__":
    main()

