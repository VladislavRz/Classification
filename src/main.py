import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')

cat_cols = ['spkts','dpkts','dbytes','proto','service',
            'state','label','sttl','dttl','dload','sloss',
            'dloss','dinpkt','sjit','djit','swin','stcpb',
            'dtcpb','dwin','tcprtt','synack','ackdat','dmean',
            'trans_depth','response_body_len','ct_state_ttl',
            'is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','is_sm_ips_ports',
            'label']


def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    plt.show()


def scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns =cols)
    return std_df


def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe['label'] == 0, "label"] = 0
    dataframe.loc[dataframe['label'] != 0, "label"] = 1

    dataframe = pd.get_dummies(dataframe, columns=['proto', 'service', 'state'])
    return dataframe


def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))

    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))

    kernal_evals = dict()
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print(f"Точность обучения {name} {train_accuracy*100:.2f}  Точность теста {name} {test_accuracy*100:.2f}")
    print(f"Точность предсказания {name} {train_precision*100:.2f}  Точность предсказания на тесте {name} {test_precision*100:.2f}")
    print(f"Полнота обучения {name} {train_recall*100:.2f}  Полнота на тесте {name} {test_recall*100:.2f}")


    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)


if __name__ == '__main__':
    data_train = pd.read_parquet(r'res/data3.parquet')
    # print(data_train.info())
    # print(data_train.head())
    # data_train['label'] = data_train['label'].astype(str)
    # data_train.loc[data_train['label'] == "normal", "label"] = 'normal'
    # data_train.loc[data_train['label'] != 'normal', "label"] = 'attack'

    scaled_train = preprocess(data_train)
    x = scaled_train.drop(['label'], axis=1).values
    y = scaled_train['label'].values
    y = y.astype('int')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=42)

    lr = LogisticRegression().fit(x_train, y_train)
    evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)

    gnb = GaussianNB().fit(x_train, y_train)
    evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)

    rf = RandomForestClassifier().fit(x_train, y_train)
    evaluate_classification(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)
