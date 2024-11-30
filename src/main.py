import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

warnings.filterwarnings('ignore')
cat_cols = ['spkts', 'dpkts', 'dbytes', 'proto', 'service',
            'state', 'label', 'sttl', 'dttl', 'dload', 'sloss',
            'dloss', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
            'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'dmean',
            'trans_depth', 'response_body_len', 'ct_state_ttl',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports',
            'label']
TRAIN_SIZE = 0.67
RANDOM_STATE = 42
TARGET = 'label'


def scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df


def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = scaling(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe[TARGET] == 0, TARGET] = 0
    dataframe.loc[dataframe[TARGET] != 0, TARGET] = 1

    dataframe = pd.get_dummies(dataframe, columns=['proto', 'service', 'state'])
    return dataframe


def evaluate_classification(model, name, test_data, test_values):
    test_precision = metrics.precision_score(test_values, model.predict(test_data))
    print(f"Значение precision для {name} на тестовой выборке: {test_precision*100:.2f}")


def print_state(train_time, cpu_percentage):
    print(f"Время, затраченное на тренировку модели: {train_time:.3f} c")
    print(f"Загрузка процессора во время тренировки: {cpu_percentage} %\n")


def split_data(data):
    scaled_train = preprocess(data_train)
    x = scaled_train.drop([TARGET], axis=1).values
    y = scaled_train[TARGET].values
    y = y.astype('int')
    return train_test_split(x, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)


def benchmark(func):
    import time
    from psutil import cpu_percent

    def wrapper(*args, **kwargs):
        start = time.time()

        ret_value = func(*args, **kwargs)

        end = time.time()
        train_time = end - start
        cpu_percentage = cpu_percent(interval=train_time)

        # Вывод характеристик процессора
        print_state(train_time, cpu_percentage)

        return ret_value

    return wrapper


@benchmark
def train_model(model, name, train_data, train_target):
    print(f"Начало тренировки модели {name}")
    return model.fit(train_data, train_target)


if __name__ == '__main__':
    lr_name = "Logistic Regression"
    gnb_name = "GaussianNB"
    rf_name = "RandomForestClassifier"

    # Подготовка данных
    data_train = pd.read_parquet(r'res/data3.parquet')
    x_train, x_test, y_train, y_test = split_data(data_train)

    # Установка различных праметров для поиска наилучших
    lr = GridSearchCV(LogisticRegression(), {'max_iter': [100, 200], 'C': [1.0, 0.3]})
    gnb = GridSearchCV(GaussianNB(),  {'var_smoothing': np.logspace(0, -9, num=100),
                                       'priors': [0.1, 0.9]})

    # Тренировка моделей
    lr = train_model(lr, lr_name, x_train, y_train)
    gnb = train_model(GaussianNB(), gnb_name, x_train, y_train)
    rf = train_model(RandomForestClassifier(), rf_name, x_train, y_train)

    # Вывод результатов
    evaluate_classification(lr, lr_name, x_test, y_test)
    evaluate_classification(gnb, gnb_name, x_test, y_test)
    evaluate_classification(rf, rf_name, x_test, y_test)
