'''
CSCI 720 Final Project

@author: Jingyang Li

@instructor: Thomas Kinsman
'''
import pandas as pd
import final_visualization as fv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance


def prepare():
    '''
    data preparation
    :return: dataframe
    '''
    df = pd.read_csv('xAPI-Edu-Data.csv')
    # drop all the IDs
    df.drop(['StageID','GradeID','SectionID'],inplace=True, axis=1)
    # replace "KW" with "KuwaIT"
    df.NationalITy.replace('KW','KuwaIT',inplace=True)
    return df

def immigrant(df):
    '''
    add new feature "isImmigrant" to dataframe
    :param df: dataframe
    :return: dataframe with new feature
    '''
    # find the ones with the same nationality and placeofbirth
    is_native = df.index[(df.NationalITy == df.PlaceofBirth)]

    df['isImmigrant'] = pd.Series(True,index=df.index)

    # set native people to false
    df.set_value(is_native,'isImmigrant',False)

    return df



def analysis(df):
    '''
    do all kinds of analysis
    :param df: dataframe
    :return: None
    '''
    # set target variable and features
    target = df['Class']
    features = df.drop(['Class'],axis=1)
    focus_features = df[['VisITedResources','Discussion','AnnouncementsView','raisedhands']]

    # convert categorical data into numerical data
    label = preprocessing.LabelEncoder()

    # find the all categorical features
    cat_cols = features.dtypes.pipe(lambda features: features[features=='object']).index
    for col in cat_cols:
        features[col] =label.fit_transform(features[col])

    # plot the correlation heatmap
    sns.heatmap(features.corr())
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title("Correlation Matrix")
    sns.plt.show()

    # use five fold cross validation
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=42)

    keys = []
    scores = []

    # initialize all models
    models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(n_estimators=300,random_state=42), 'XGBoost': XGBClassifier(seed=42),\
              'AdaBoost' : AdaBoostClassifier()}

    # iteratively test each model and show results
    for key,value in models.items():
        model = value
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print('Results for: ' + str(key) + '\n')
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
        acc = accuracy_score(y_test, pred)
        print("accuracy is "+ str(acc))
        print('\n' + '\n')
        keys.append(key)
        scores.append(acc)
        table = pd.DataFrame({'model':keys, 'accuracy score':scores})

    print(table)

    xgb = XGBClassifier(seed=42)
    pred = xgb.fit(X_train, y_train).predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print("accuracy is " + str(accuracy_score(y_test, pred)))
    plot_importance(xgb)
    plt.show()


    d_values = []
    l_values = []
    n_values = []
    acc_values = []
    # define the range of three parameters
    depth = [3, 4]
    learning_Rate = [0.1, 0.5, 1]
    n_estimators = [50, 100, 150, 200]

    # iteratively apply XGBoost with different combinations of parameters
    for d in depth:
        for l in learning_Rate:
            for n in n_estimators:
                new_xgb = XGBClassifier(max_depth=d, learning_rate=l, n_estimators=n, seed=42)
                pred = new_xgb.fit(X_train, y_train).predict(X_test)
                acc = accuracy_score(y_test, pred)
                d_values.append(d)
                l_values.append(l)
                n_values.append(n)
                acc_values.append(acc)

    dict = {'max_depth': d_values, 'learning_rate': l_values, 'n_estimators': n_values,
            'accuracy': acc_values}

    # build a dataframe to show the comparison
    output = pd.DataFrame.from_dict(data=dict)
    print(output.sort_values(by='accuracy', ascending=False))


if __name__ == '__main__':

    data = prepare()
    data = immigrant(data)
    focus_features = data[['VisITedResources','Discussion','AnnouncementsView','raisedhands','Class']]
    cat_attrs = ['gender','NationalITy','PlaceofBirth','StageID','Relation','GradeID','Topic','Semester','StudentAbsenceDays']
    num_attrs = ['raisedhands', 'VisITedResources', 'AnnouncementsView','Discussion','Class']
    analysis(data)
    # plot categorical attributes
    fv.plot_cat(data,cat_attrs)
    # plot numerical attributes
    fv.plot_num(data,num_attrs)
    # pair comparison of top four features
    fv.pair_grid(focus_features)
    # fv.immigrants(data)

