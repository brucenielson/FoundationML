from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def drop_columns(data, col_list):
    for col in col_list:
        data = data.loc[:,data.columns != col]

    return data

def rename_column(data, old_name, new_name):
    data = data.rename(columns = {old_name:new_name})
    return data



def create_train_test():
    # Get donation data
    datafile = os.getcwd() + '\\DonationData.csv'
    donation_data = pd.read_csv(datafile, dtype={'Alias':int})
    donation_data['Donated'] = (donation_data['FirstDonationAmount'] > 0.0)

    # donation_data = drop_columns(donation_data, ['M6Date','FirstDate','FirstState','M6State','Alias','M6Rank','M6TeneureAtRank','M6TenureMonths','PersonalEnrollments_M6Date','FirstDonationAmount'])
    # donation_data = rename_column(donation_data, 'FirstDRank', 'Rank')
    # donation_data = rename_column(donation_data, 'FirstTenureAtRank', 'TenureAtRank')
    # donation_data = rename_column(donation_data, 'FirstTenureMonths', 'TenureAtMonths')
    # donation_data = rename_column(donation_data, 'PersonalEnrollments_FirstDate', 'PersonalEnrollments')

    donation_data = drop_columns(donation_data, ['M6Date','FirstDate','FirstState','M6State','Alias','FirstDRank','FirstTenureAtRank','FirstTenureMonths','PersonalEnrollments_FirstDate','FirstDonationAmount'])
    donation_data = rename_column(donation_data, 'M6Rank', 'Rank')
    donation_data = rename_column(donation_data, 'M6TeneureAtRank', 'TenureAtRank')
    donation_data = rename_column(donation_data, 'M6TenureMonths', 'TenureAtMonths')
    donation_data = rename_column(donation_data, 'PersonalEnrollments_M6Date', 'PersonalEnrollments')


    # Get control data
    datafile = os.getcwd() + '\\ControlData.csv'
    control_data = pd.read_csv(datafile, dtype={'Alias':int})
    control_data['Donated'] = False
    control_data = drop_columns(control_data, ['State','Alias'])

    # Combine data
    all_data = pd.concat([donation_data, control_data], axis=0)

    train, test = train_test_split(all_data,test_size=0.25)
    y_train = train['Donated']
    X_train = train.loc[:, train.columns != 'Donated']
    X_train = X_train.loc[:, X_train.columns != 'FirstDonationAmount']

    y_test = test['Donated']
    X_test = test.loc[:, test.columns != 'Donated']
    X_test = X_test.loc[:, X_test.columns != 'FirstDonationAmount']
    return X_train, y_train, X_test, y_test


def train(X, y):
    clf = LogisticRegression(penalty='l2', C=10.0)
    clf.fit(X, y)
    return clf


def print_stats(name, X, y, y_pred):
    # returns statistics
    print("")
    print("Results of Predict - "+name+":")
    print('Total Samples:', len(X))
    print('Misclassified train samples: %d' % (y != y_pred).sum())
    print('Accuracy of train set: %.4f' % accuracy_score(y, y_pred))
    print('')
    print('')


def print_correlations(X, y):
    # Correlations
    X_corr = X.copy(deep=True)
    X_corr['Donated'] = y.copy(deep=True)
    correlations = X_corr.corr()['Donated'].to_frame()
    correlations['Correlation'] = abs(correlations['Donated'])
    correlations = correlations.sort_values('Correlation', ascending=False)
    correlations = correlations.drop(['Donated'], axis=0)
    correlations = correlations.drop(['Donated'], axis=1)
    print(correlations)


def set_prediction(data, probability):
    data['Prediction'] = data['ProbTrue'] >= probability
    return data


def stats(clf, X, y):
    # Predict
    y_pred = clf.predict(X)
    print_stats("Training Set", X, y, y_pred)

    # Default Predict
    size = len(y_pred)
    y_default = [False]*size
    print_stats("Default Set", X, y, y_default)

    # Correlations
    print_correlations(X, y)

    # Classification Report
    print(classification_report(y, y_pred))

    # Create one large dataframe to play with that has both ground truth and predictions
    y_pred = pd.DataFrame(y_pred, columns=['Prediction'])
    y = pd.DataFrame(y)
    all_data = pd.concat([X, y], axis=1)
    all_data.reset_index(drop=True, inplace=True)
    all_data = pd.concat([all_data, y_pred], axis=1)
    y_pred_prob = clf.predict_proba(X)
    y_pred_prob = pd.DataFrame(y_pred_prob, columns=['ProbFalse', "ProbTrue"])
    all_data = pd.concat([all_data, y_pred_prob['ProbTrue']], axis=1)
    # print(all_data.head(5))


    # How accurate are the positive predictions?
    true_predictions = all_data[all_data['Prediction'] == True]
    y_positive_pred = true_predictions['Prediction']
    y_positive = true_predictions['Donated']
    X_postitive = true_predictions.drop(['Prediction'], axis=1)
    X_postitive = X_postitive.drop(['Donated'], axis=1)

    print_stats("Positive Predictions 50%", X_postitive, y_positive, y_positive_pred)

    all_data = set_prediction(all_data, 0.10)
    # How accurate are the positive predictions?
    true_predictions = all_data[all_data['Prediction'] == True]
    y_positive_pred = true_predictions['Prediction']
    y_positive = true_predictions['Donated']
    X_postitive = true_predictions.drop(['Prediction'], axis=1)
    X_postitive = X_postitive.drop(['Donated'], axis=1)

    print_stats("Positive Predictions 10%", X_postitive, y_positive, y_positive_pred)





X_train, y_train, X_test, y_test = create_train_test()
clf = train(X_train, y_train)
stats(clf, X_train, y_train)
stats(clf, X_test, y_test)
