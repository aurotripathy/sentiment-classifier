
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data():
    # Load tweets and assign labels
    train_val_complaint = pd.read_csv('data/complaint1700.csv')
    train_val_complaint['label'] = 0
    train_val_non_complaint = pd.read_csv('data/noncomplaint1700.csv')
    train_val_non_complaint['label'] = 1

    # Concatenate complaint and non-complaint tweets
    train_val_all = pd.concat([train_val_complaint, train_val_non_complaint], axis=0).reset_index(drop=True)

    X = train_val_all.tweet.values
    y = train_val_all.label.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)

    # Drop 'airline' column
    train_val_all.drop(['airline'], inplace=True, axis=1)

    # Display 5 random samples
    print("Displaying train samples...")
    print(train_val_all.sample(5))

    # Load test data
    test_data = pd.read_csv('data/test_data.csv')

    # Keep important columns
    test_data = test_data[['id', 'tweet']]
    return X, y, X_train, X_val, y_train, y_val, train_val_all, test_data

