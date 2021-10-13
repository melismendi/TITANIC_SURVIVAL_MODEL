# İş Problemi:Özellikleri verildiğinde insanların hayatta kalmasını tahmin edebilir misiniz?
#1:Kişinin hayatta kalmasını, 0 ise hayatını kaybetmesini temsil etmektedir.("Survived" değişkeni)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from helpers.eda import *
from helpers.data_prep import *

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

#veri ön işleme ve değişken mühendisliği için kullanacağımız fonksiyonlar.
from helpers.eda import *
from helpers.data_prep import *
df = pd.read_csv("Hafta6/titanic.csv")
df.head()
#Veri setine önceki 6.Haftada yaptığımız veri önişleme ve değişken müh. işlemlerini uygulayan fonksiyon:
def titanic_data_prep(dataframe):
    from sklearn.preprocessing import StandardScaler

    dataframe.columns = [col.upper() for col in dataframe.columns]
    #feature eng : değişken üretmek
    # Cabin bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    # Name count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    # name word count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    # age_pclass
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    # is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    # features' types : kategorik,nümerik,kardinal değişkenleri görmek
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # missings
    dataframe.drop("CABIN", inplace=True, axis=1)
    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    # feature eng
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]

    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # label encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # rare encoding
    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    # one hot encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    # features' types
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # useless features
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                    (dataframe[col].value_counts() / len(dataframe) < 0.01).any(axis=None)]
    dataframe.drop(useless_cols, axis=1, inplace=True)

    # scaling
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe

df=titanic_data_prep(df)
df.head()

# Bağımlı ve bağımsız değişkelerin seçip model kuralım:
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

# Model:
log_model = LogisticRegression().fit(X, y)

#b ve w ağırlıklarının çıktısını almak istersek: (y=b+w1x1+w2x2+...+wpxp)
log_model.intercept_ #byi verir. array([-0.52351966])
log_model.intercept_[0] #sayi formatinda görmek istersek -0.5235196623000302
log_model.coef_  #w ağırlıklarını.

#Tahmin
y_pred=log_model.predict(X)
y_pred[0:10]
y[0:10]

#Modelin başarısını değerlendirmek istersek:
# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
plot_confusion_matrix(y,y_pred) #grafikte gördüğüm accuracy score:0.84

#Bütün başarı skorlarını bastırmak istersek:
print(classification_report(y, y_pred)) #f1 score:0.87 yuzde87 demek.
# Bunlara baktık çünkü bazen accuracy yeterli olmaz. Diğer metriklere bakmak gerekebilir.

# ROC AUC:
y_prob = log_model.predict_proba(X)[:, 1] #içine parametre olarak olasılık ister.
roc_auc_score(y, y_prob) #çıktı:0.8849316673590472


# Model Validation: Holdout
# Holdout Yöntemi
# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)
# Modelin train setine kurulması:
log_model = LogisticRegression().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)


# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]
# Classification report
print(classification_report(y_test, y_pred))  #f1-score: 0.82

# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

#graf. gördüğümüz AUC: 0.86
roc_auc_score(y_test, y_prob)  #çıktı:0.8623675368312226











