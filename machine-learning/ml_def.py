#%%
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import re
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

sns.set()
warnings.filterwarnings("ignore")

#%%
# under construction 

# def quick_fit(df,target, seed ):
    
    
    
    
    
    
    #X, y = df
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=df[target])
    #pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

    #pipe.fit(X_train, y_train)

    #pipe.score(X_test, y_test)
    
    #Xtrain_list.append(X_train)
    #Xtest_list.append(X_test)
    #ytrain_list.append(y_train)
    #ytest_list.append(y_test)

#%%

# Source: https://www.kaggle.com/gifarihoque/pidd-missing-data-ml-iterimputer-tut-86

def gimmeThemStats(dFrame):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap, and outputs a distribution plot.
    
    Parameters
    ----
    dFrame(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    Nothing.
    
    """
    # Description
    print("Descriptive Stats:")
    display(dFrame.describe().T)
    
    # Heatmap
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(dFrame.corr(), annot =True)
    
    # Distribution
    ### NOTE: I changed histplot to distplot
    fig, axes = plt.subplots(4, 2, figsize=(14,14))
    fig.suptitle("Distribution Plot", y=0.92, fontsize='x-large')
    fig.tight_layout(pad=4.0)

    for i,j in enumerate(df.columns[:-1]):
        sns.distplot(dFrame[j], ax=axes[i//2, i%2])


#%%        
def imputeEm(adf, estimList, stylesList):
    """
    Description
    ----
    Iteratively imputes missing values to a dataset by following a 
    given estimator and imputation-order pair.
    If an estimator or imputation order throws an error, an error
    will be printed after function call explaining why the error
    occured. If no error is found for a given estimator or imputation
    order, no errors will be printed in the end of the function call.
    An error thrown for an estimator and imputation order pair won't 
    affect other pairs. You'll still get the results you sought for.
    
    Parameters
    ----
    adf (dataframe):
        The dataframe containing missing values.
        
    estimList (list of models):
        The list of estimators to use in IterativeImputer.
    
    stylesList (list of str):
        The list of styles to use in IterativeImputer.
    
    Returns
    ----
    estim_name_list (list of str):
        A list of the name of the estimator used in each iteration.
        For example, if there are 5 imputation-order styles per each 
        estimator, then the list will contain each estimator 5 times.
    
    style_list (list of str):
        A list of the name of the imputation-order used in each
        iteration. For example, if there are 10 estimators used, 
        the list will include each imputation-order 10 times.
        
    imputed_df_list (list of dataframes):
        A list of dataframes for each estimator and imputation-order
        pair.
    """

    # The returned lists
    style_list = []
    estim_name_list = []
    imputed_df_list = []
    
    # Loop through each estimator
    for estim in range(len(estimList)):
        
        # Convert estimator to string format and debolish parenthesis and anything in between
        estimstorName = re.sub(r"\([^()]*\)", '', str(estimList[estim])) 
        
        # Loop through each imputation-order
        for style in stylesList:
            
            try:
                # Introduce Iterative Imputer with estimator and imputation_order
                imputer = IterativeImputer(random_state=42, estimator=estimList[estim], imputation_order=style)
                # Fit on dataframe
                imputer.fit(adf)
            except Exception as e:
                print("==============================================================")
                print(f"I wasn't able to iteratively impute with the estimator: {estimList[estim]} and imputation order: {style}.")
                print(f"This was the error I've received from my master:\n\n{e}.")
                print("\nI didn't let it faze me though, for now I've skipped this imputation pair.")
                print("==============================================================\n")
            else:
                estim_name_list.append(estimstorName) #Appending estimator name
                style_list.append(style) #Appending style name
                
                # Transform and append the imputed dataframe to the list of imputed dataframes
                imputed_df_list.append(pd.DataFrame(imputer.transform(adf), columns = adf.columns))
            
            
    return estim_name_list, style_list, imputed_df_list

#%%
def invalidNumberChecker(dataList):
    """
    Description
    ----
    This function will check for values less than or equal to 0 within a dataframe. 
    The function displays several things including: 
        1. The Dataframe number: The ith dataframe in `dataList[2][i]` in which the 
        invalid number was caught.
        2. Estimator: Estimator used in the dataframe number.
        3. Order: Imputation order used in the dataframe number.
        4. A set of description for each invalid number within the dataframe which 
        includes:
            4a. Index: The index of the dataframe where the invalid number lives.
            4b. Column: The column that contains the invalid number.
            4c. Value: The invalid number itself.
        5. A dataframe display of the rows with invalid number(s).
    
    Parameters
    ----
    dataList (list of lists):
        The list containing a list of models, list of imputation orders, and
        list of dataframes which was obtained after running function `imputeEm`.
    
    Returns
    ----
    Nothing.
    
    """
    # Loop through every dataframe in the list of dataframes
    for i in range(len(dataList[2])):
        # index_list will hold the indices where invalid numbers live
        index_list = []
        # invalid_pairs is a list containing pairs (tuples) of rows and column names where invalid numbers live
        invalid_pairs = dataList[2][i][colsToFix][dataList[2][i][colsToFix] <= 0].stack().index.tolist()
        if(invalid_pairs):
            print(f'Dataframe # {i}  --  For reference, check dataList[2][{i}], where dataList is the list obtained after running function `imputeEm`.')
            print('--------------')
            print(f'Estimator: {dataList[0][i]}\nOrder: {dataList[1][i]}\n')
            for j in range(len(invalid_pairs)):
                index_list.append(invalid_pairs[j][0])
                print(f'Index: {invalid_pairs[j][0]}\nColumn: {invalid_pairs[j][1]}')
                print(f'Value: {dataList[2][i][invalid_pairs[j][1]].loc[invalid_pairs[j][0]]}\n')
            display(dataList[2][i].loc[index_list])
            print("==================================================================================================\n\n")
            
#%%
def produceSplits(dataList, testSize=0.2):
    """
    Description
    ----
    Splits a list of dataframes into train and test sets based
    on given testSize for train_test_split.
    For each dataframe in dfList, the X_train, X_test, y_train,
    and y_test are appended into separate lists.
    Each split will use the same index for all dataframes in 
    dataList[2] to reduce bias when comparing results after 
    modeling.
    
    Parameters
    ----
    dataList (list of dataframes):
        The list containing a list of models, list of imputation 
        orders, and list of dataframes which was obtained after 
        running function `imputeEm`.
    
    testSize (float):
        The test_size that you want to give for train_test_split.
        The default test_size is set to 0.2.
        
    Returns
    ----
    Xtrain_list (list of dataframes):
        A list containing the X_train split for each dataframe
        in dfList.
    
    Xtest_list (list of dataframes):
        A list containing the X_test split for each dataframe
        in dfList.
        
    ytrain_list (list of series):
        A list containing the y_train split for each dataframe
        in dfList.
        
    ytest_list (list of series):
        A list containing the y_test split for each dataframe
        in dfList.
    
    """
    
    # Returned train and test splits lists
    Xtrain_list = []
    Xtest_list = []
    ytrain_list = []
    ytest_list = []
    
    # Loop through each dataframe in dataList[2]
    for dFrame in range(len(dataList[2])):
        # Inputs
        X = dataList[2][dFrame].drop('survived', axis=1)
        # Output
        y = dataList[2][dFrame]['survived']
        # Train and test split with given testSize, where the default is 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42, stratify = y)
        # Append the splits to each list
        Xtrain_list.append(X_train)
        Xtest_list.append(X_test)
        ytrain_list.append(y_train)
        ytest_list.append(y_test)
    
    return Xtrain_list, Xtest_list, ytrain_list, ytest_list

#%%
def weightForMe(trainList, testList):
    """
    Description
    ----
    Standardizes the training and testing input sets 
    using StandardScaler().
    The training features are fit and then transformed.
    The testing features are transformed.
    
    Parameters
    ----
    trainList (list of dataframes):
        The list of dataframes of the training set obtained
        after running function `produceSplits`.

    testList (list of dataframes):
        The list of dataframes of the testing set obtained
        after running function `produceSplits`.
        
    Returns
    ----
    list_of_scaled_train_dfs (list of dataframes):
        List of dataframes of each `X_train` after scale.
        
    list_of_scaled_test_dfs (list of dataframes):
        List of dataframes of each `X_test` after scale.
    """
    # Returned lists
    list_of_scaled_train_dfs = []
    list_of_scaled_test_dfs = []
    
    # Iterate through each `X_train` and `X_test` in `trainList`
    for i in range(len(trainList)):
        # Introducing the Scaler
        sclr = StandardScaler()
        
        scaled_train_features = sclr.fit_transform(trainList[i]) # fit and transform train set
        scaled_test_features = sclr.transform(testList[i]) # transform test set
            
        # For debugging purposes, I converted the scaled lists to dataframes
        list_of_scaled_train_dfs.append(pd.DataFrame(scaled_train_features, 
                                                 index = trainList[i].index, 
                                                 columns = trainList[i].columns))
        list_of_scaled_test_dfs.append(pd.DataFrame(scaled_test_features, 
                                                 index = testList[i].index, 
                                                 columns = testList[i].columns))
                                   
    return list_of_scaled_train_dfs, list_of_scaled_test_dfs

#%%
def acceptingModels(XTrainList, XTestList, yTrainList, yTestList, dataList, classifierList):
    """
    Description
    ----
    This function tests out all models listed in `classifierList`.
    If the model isn't valid, the function prints out the invalid
    name of the model along with it's error.
    The function then fits the model to every train set in
    `XTrainList` and `yTrainList` and gives a prediction based on
    the `XTestList`. The accuracy score is then found along with
    the diabetic positive precision, recall and f-scores, and the
    diabetic negative prevision, recall and f-scores.
    Data is then appended into a dictionary with columns:
        1.  ModelName - Name of model used for predictions.
        2.  Estimator - Name of estimator used for iterative
              imputation.
        3.  Order - The type of imputation order style used.
        4.  AccuracyScore - The accuracy score of the model's
              predictions.
        5.  CorrectPredictionsCount - The number of predictions
              that the model got correct.
        6.  Total - The size of XTestList; the total number of
              patients in the test set.
        7.  PosPrecScore - The precision score for diabetic
              positives.
        8.  PosRecScore - The recall score for diabetic positives.
        9.  PosFScore - The f1-score for diabetic positives.
        10. NegPresScore - The precision score for diabetic
              negatives.
        11. NegRecScore - The recall score for diabetic negatives.
        12. NegFScore - The f1-score for diabetic negatives.
        13. TNPercentage - The ratio of True Negatives to total
              (The [0][0] index of confusion matrix).
        14. TPPercentage - The ratio of True Positives to total
              (The [1][1] index of confusion matrix).
        15. FNPercentage - The ratio of False Negatives to total
              (The [1][0] index of confusion matrix).
        16. FPPercentage - The ratio of False Positives to total
              (The [0][1] index of confusion matrix).
    NOTE: 
        TNPercentage + TPPercentage + 
            FNPercentage + FPPercentage = 100.
    
    Parameters
    ----
    XTrainList (list of dataframes):
        A list containing the X_train split for each dataframe.
        
    XTestList (list of dataframes):
        A list containing the X_test split for each dataframe.
        
    yTrainList (list of series):
        A list containing the y_train split for each dataframe.
        
    yTestList (list of series):
        A list containing the y_test split for each dataframe.
    
    dataList (list of lists):
        The list containing a list of models, list of imputation 
        orders, and list of dataframes which was obtained after 
        running function `imputeEm`.
        
    classifierList (list of models):
        The list of models chosen.
    
    Returns
    ----
    dic (dataframe):
        A dataframe of dic.
        
    """
    # Introduce a dictionary
    dic = {'ModelName': [], 'Estimator': [], 'Order': [], 'AccuracyScore':[], 
           'CorrectPredictionsCount': [], 'Total': [], 'PosPrecScore': [],
           'PosRecScore': [], 'PosFScore': [], 'NegPrecScore': [], 'NegRecScore': [],
           'NegFScore': [], 'TNPercentage': [], 'TPPercentage': [], 
           'FNPercentage': [], 'FPPercentage': []}
    
    # Deepcopy the classifierList
    models = deepcopy(classifierList)
    
    # Test each models in the list to verify validation
    for i in range(len(classifierList)):
        try:
            model = classifierList[i]
            model.fit(XTrainList[0], yTrainList[0])
        except Exception as e:
            print("==============================================================")
            print(f"I wasn't able to score with the model: {classifications_list[i]}")
            print(f"This was the error I've received from my master:\n\n{e}.")
            print("\nI didn't let it faze me though, for now I've skipped this model.")
            print("==============================================================\n")
            models.remove(classifierList[i]) # Remove invalid models from list
    
    # Loop through all train/test sets
    for i in range(len(XTrainList)):
        # Loop through all models
        for classifier in range(len(models)):
            # Destroy parenthesis and anything within
            modelName = re.sub(r"\([^()]*\)", '', str(models[classifier]))
            # Performance
            model = models[classifier]
            model.fit(XTrainList[i], yTrainList[i])          
            pred = model.predict(XTestList[i])
            # Results
            acc_score = accuracy_score(yTestList[i], pred)
            noOfCorrect = accuracy_score(yTestList[i], pred, normalize = False)
            total = noOfCorrect/acc_score
            madConfusing = confusion_matrix(yTestList[i],pred)

            dpps = madConfusing[1][1] / (madConfusing[1][1] + madConfusing[0][1]) # diab pos prec score
            dprs = madConfusing[1][1] / (madConfusing[1][1] + madConfusing[1][0]) # diab pos rec score
            dpfs = 2 * (dpps * dprs) / (dpps + dprs) # diab pos f1 score
            dnps = madConfusing[0][0] / (madConfusing[0][0] + madConfusing[1][0]) # diabetic neg prec score
            dnrs = madConfusing[0][0] / (madConfusing[0][0] + madConfusing[0][1]) # diab neg rec score
            dnfs = 2 * (dnps * dnrs) / (dnps + dnrs) # diab neg f1 score
            
            # Save everything
            dic['ModelName'].append(modelName)
            dic['Estimator'].append(dataList[0][i])
            dic['Order'].append(dataList[1][i])
            dic['AccuracyScore'].append(acc_score)
            dic['CorrectPredictionsCount'].append(noOfCorrect)
            dic['Total'].append(total)
            dic['PosPrecScore'].append(dpps)
            dic['PosRecScore'].append(dprs)
            dic['PosFScore'].append(dpfs)
            dic['NegPrecScore'].append(dnps)
            dic['NegRecScore'].append(dnrs)
            dic['NegFScore'].append(dnfs)
            dic['TNPercentage'].append(madConfusing[0][0]/total*100)
            dic['TPPercentage'].append(madConfusing[1][1]/total*100)
            dic['FNPercentage'].append(madConfusing[1][0]/total*100)
            dic['FPPercentage'].append(madConfusing[0][1]/total*100)
            
    return pd.DataFrame.from_dict(dic)

#%%
def acceptingModelsNoIterImp(XTrain, XTest, yTrain, yTest, classifierList):
    """
    Description
    ----
    This function tests out all models listed in `classifierList`
    without Iterative Imputation.
    If the model isn't valid, the function prints out the invalid
    name of the model along with it's error.
    The function then fits the model to the train set and gives a 
    prediction based on the `XTest`. The accuracy score is then 
    found along with the diabetic positive precision, recall and 
    f-scores, and the diabetic negative prevision, recall and 
    f-scores.
    
    Data is then appended into a dictionary with columns:
        1.  ModelName - Name of model used for predictions.
        2.  AccuracyScore - The accuracy score of the model's
              predictions.
        3.  CorrectPredictionsCount - The number of predictions
              that the model got correct.
        4.  Total - The size of XTestList; the total number of
              patients in the test set.
        5.  PosPrecScore - The precision score for diabetic
              positives.
        6.  PosRecScore - The recall score for diabetic positives.
        7.  PosFScore - The f1-score for diabetic positives.
        8.  NegPresScore - The precision score for diabetic
              negatives.
        9.  NegRecScore - The recall score for diabetic negatives.
        10. NegFScore - The f1-score for diabetic negatives.
        11. TNPercentage - The ratio of True Negatives to total
              (The [0][0] index of confusion matrix).
        12. TPPercentage - The ratio of True Positives to total
              (The [1][1] index of confusion matrix).
        13. FNPercentage - The ratio of False Negatives to total
              (The [1][0] index of confusion matrix).
        14. FPPercentage - The ratio of False Positives to total
              (The [0][1] index of confusion matrix).
    NOTE: 
        TNPercentage + TPPercentage + 
            FNPercentage + FPPercentage = 100.
    
    Parameters
    ----
    XTrain (dataframe/series):
        The X_train split for the dataframe.
        
    XTest (dataframe/series):
        The X_test split for the dataframe.
        
    yTrain (series):
        The y_train split for the dataframe.
        
    yTest (series):
        The y_test split for the dataframe.
        
    classifierList (list of models):
        The list of models chosen.
    
    Returns
    ----
    dic (dataframe):
        A dataframe of dic.
        
    """
    # Introduce a dictionary
    dic = {'ModelName': [], 'AccuracyScore':[],
           'CorrectPredictionsCount': [], 'Total': [], 'PosPrecScore': [],
           'PosRecScore': [], 'PosFScore': [], 'NegPrecScore': [], 'NegRecScore': [],
           'NegFScore': [], 'TNPercentage': [], 'TPPercentage': [], 
           'FNPercentage': [], 'FPPercentage': []}
    
    # Deepcopy the classifierList
    models = deepcopy(classifierList)
    
    # Test each models in the list to verify validation
    for i in range(len(classifierList)):
        try:
            model = classifierList[i]
            model.fit(XTrain, yTrain)
        except Exception as e:
            print("==============================================================")
            print(f"I wasn't able to score with the model: {classifications_list[i]}")
            print(f"This was the error I've received from my master:\n\n{e}.")
            print("\nI didn't let it faze me though, for now I've skipped this model.")
            print("==============================================================\n")
            models.remove(classifierList[i]) # Remove invalid models from list
    
    # Loop through all models
    for classifier in range(len(models)):
        # Destroy parenthesis and anything within
        modelName = re.sub(r"\([^()]*\)", '', str(models[classifier]))
        # Performance
        model = models[classifier]
        model.fit(XTrain, yTrain)          
        pred = model.predict(XTest)
        # Results
        acc_score = accuracy_score(yTest, pred)
        noOfCorrect = accuracy_score(yTest, pred, normalize = False)
        total = noOfCorrect/acc_score
        madConfusing = confusion_matrix(yTest,pred)

        dpps = madConfusing[1][1] / (madConfusing[1][1] + madConfusing[0][1]) # diab pos prec score
        dprs = madConfusing[1][1] / (madConfusing[1][1] + madConfusing[1][0]) # diab pos rec score
        dpfs = 2 * (dpps * dprs) / (dpps + dprs) # diab pos f1 score
        dnps = madConfusing[0][0] / (madConfusing[0][0] + madConfusing[1][0]) # diabetic neg prec score
        dnrs = madConfusing[0][0] / (madConfusing[0][0] + madConfusing[0][1]) # diab neg rec score
        dnfs = 2 * (dnps * dnrs) / (dnps + dnrs) # diab neg f1 score

        # Save everything
        dic['ModelName'].append(modelName)
        dic['AccuracyScore'].append(acc_score)
        dic['CorrectPredictionsCount'].append(noOfCorrect)
        dic['Total'].append(total)
        dic['PosPrecScore'].append(dpps)
        dic['PosRecScore'].append(dprs)
        dic['PosFScore'].append(dpfs)
        dic['NegPrecScore'].append(dnps)
        dic['NegRecScore'].append(dnrs)
        dic['NegFScore'].append(dnfs)
        dic['TNPercentage'].append(madConfusing[0][0]/total*100)
        dic['TPPercentage'].append(madConfusing[1][1]/total*100)
        dic['FNPercentage'].append(madConfusing[1][0]/total*100)
        dic['FPPercentage'].append(madConfusing[0][1]/total*100)
            
    return pd.DataFrame.from_dict(dic)

#%%
def magicWithoutIterImp(adf, testSize = 0.2):
    """
    Description
    ----
    Splits and scales a single given dataframe using
    `StandardScaler()`. The scaled features are then
    inputted into `acceptingModelsNoIterImp` and out
    comes a dataframe.
    
    Parameters
    ----
    adf (dataframe):
        The dataframe to use for modeling.
    
    testSize (float):
        The test_size that you want to give for 
        train_test_split.
        The default test_size is set to 0.2.
    
    Returns
    ----
    save (dataframe):
        The dataframe after running the splits in
        `acceptingModelsNoIterImp`.
        
    """
    # Define X
    X = adf.drop('Outcome', axis=1)
    # Define y
    y = adf['Outcome']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 42, stratify = y)

    # Scale
    sclr = StandardScaler()
    scaled_train_features = sclr.fit_transform(X_train) 
    scaled_test_features = sclr.transform(X_test) 

    # Save
    save = acceptingModelsNoIterImp(scaled_train_features, scaled_test_features, y_train, y_test, classifications_list)
    
    return save
#%%

def finalResults(adf, saves):
    """
    Description
    ----
    Prints the overall average of the method/technique's 
    `AccuracyScore` column.
    Displays the information of the best performed model
    for the current method.
    Saves data into a dictionary only if the name of the
    dataframe `adf` does not exist in saves['MethodName'],
    and finally displays whatever data is in `saves`.
    
    Parameters
    ----
    adf (dataframe):
        The dataframe of the results of a method/technique.
        
    saves (dict):
        The dictionary containing all final results.
        
    Returns
    ----
    Nothing.
        
    """
    
    print(f"Overall Accuracy Score: {adf['AccuracyScore'].mean()}") # Overall average of method/technique's `AccuracyScore` column
    print('Current Method\'s Top Model:')
    display(adf.iloc[[adf['AccuracyScore'].idxmax()]]) 
    
    if adf.name not in saves['MethodName']:
        issaSeries = adf.iloc[adf['AccuracyScore'].idxmax()]
        saves['MethodName'].append(adf.name) # Dataframe name (Method/Technique name)
        saves['MethodAccuracy'].append(adf['AccuracyScore'].mean()) # Overall average of method/technique's `AccuracyScore` column
        saves['TopModelName'].append(issaSeries['ModelName'])  # Best performed model name
        if 'Estimator' in adf.columns:
            saves['Estimator'].append(issaSeries['Estimator']) # Estimator used for iterative imputer if applicable
            saves['Order'].append(issaSeries['Order']) # Order used for iterative imputer if applicable
        else:
            saves['Estimator'].append('None')
            saves['Order'].append('None')
        saves['ModelAccuracy'].append(issaSeries['AccuracyScore']) # Accuracy score of the best performed model
        saves['CorrectPredictionsCount'].append(issaSeries['CorrectPredictionsCount']) # Number of correct predictions
        saves['Total'].append(issaSeries['Total']) # Size of test set/total number of patients in test set
    display(saves)

    
estimatorList = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=42),
    ExtraTreesRegressor(n_estimators=10, random_state=42),
    RandomForestRegressor(criterion='mse', n_estimators=10, random_state=42),
    KNeighborsRegressor(n_neighbors=15)
]

imputation_styles = ['ascending', 'descending', 'roman', 'arabic', 'random']

classifications_list = [
    LinearSVC(C= 5.0, class_weight="balanced"), SVC(kernel='rbf'), GaussianNB(), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(), RandomForestClassifier(),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
]

colsToFix = ['deck', 'age']
# %%
"""
=========================================================
Imputing missing values with variants of IterativeImputer
=========================================================

.. currentmodule:: sklearn

The :class:`~impute.IterativeImputer` class is very flexible - it can be
used with a variety of estimators to do round-robin regression, treating every
variable as an output in turn.

In this example we compare some estimators for the purpose of missing feature
imputation with :class:`~impute.IterativeImputer`:

* :class:`~linear_model.BayesianRidge`: regularized linear regression
* :class:`~tree.DecisionTreeRegressor`: non-linear regression
* :class:`~ensemble.ExtraTreesRegressor`: similar to missForest in R
* :class:`~neighbors.KNeighborsRegressor`: comparable to other KNN
  imputation approaches

Of particular interest is the ability of
:class:`~impute.IterativeImputer` to mimic the behavior of missForest, a
popular imputation package for R. In this example, we have chosen to use
:class:`~ensemble.ExtraTreesRegressor` instead of
:class:`~ensemble.RandomForestRegressor` (as in missForest) due to its
increased speed.

Note that :class:`~neighbors.KNeighborsRegressor` is different from KNN
imputation, which learns from samples with missing values by using a distance
metric that accounts for missing values, rather than imputing them.

The goal is to compare different estimators to see which one is best for the
:class:`~impute.IterativeImputer` when using a
:class:`~linear_model.BayesianRidge` estimator on the California housing
dataset with a single value randomly removed from each row.

For this particular pattern of missing values we see that
:class:`~ensemble.ExtraTreesRegressor` and
:class:`~linear_model.BayesianRidge` give the best results.

"""
#%%
# example of proper model list for def magic and magic_model 
classifications_list = [
    LinearSVC(C= 5.0, class_weight="balanced"), SVC(kernel='rbf'), GaussianNB(), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
]
#%%
# modified to accept train, validate, test split 
def magic(df, target):
    """
    Description
    ----
    Splits a single given dataframe using 
    'train_test_split'
    
    
    Parameters
    ----
    df (dataframe):
        The dataframe to use for modeling.
    target: 
        stratify=['target'] 
    
    testSize (float):
        The test_size that you want to give for 
        train_test_split.
        The default test_size is set to 0.2 for 
        test_validate and test
        The default test_size is set to 0.2 for
        train and validate
    
    Returns
    ----
    save (dataframe):
        The dataframe after running the splits in
        `model_magic`
        
    """
    
    # Split
    train_validate, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=42, stratify=train_validate[target])
    
    X_train = train.drop(columns=['churn'])
    y_train = train.churn

    X_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    X_test = test.drop(columns=['churn'])
    y_test = test.churn
    # Save
    save = model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifications_list)
    
    return save

#%%
# modified to accept train, validate, test split 
def model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifierList):
    """
    Description
    ----
    This function tests out all models listed in classifierList.
    
    If the model isn't valid, the function prints out the invalid
    name of the model along with it's error.
    The accuracy score, confusion matric, positive precision, recall 
    and f-scores, and negative prevision, recall and f-scores.
    
    Data is then appended into a dictionary with columns.
    
    
    Parameters
    ----
    The X_train split for the dataframe.
        
    The X_test split for the dataframe.

    The X_validate split for dataframe.

    The y_validate splot for dataframe.

    The y_train split for the dataframe.
        
    The y_test split for the dataframe.
        
    classifierList (list of models):
        The list of models chosen.
    
    Returns
    ----
    dic (dataframe):
        A dataframe of dic.
        
    """
    # Introduce a dictionary
    dic = {'ModelName': [], 'AccuracyScore': [], 'AccuracyScoreVAL': [],
           'CorrectPredictionsCount': [], 'CorrectPredictionsCountVAL': [], 'Total': [], 'TotalVAL': [], 
           'PosPrecScore': [], 'PosPrecScoreVAL':[], 'PosRecScore': [], 'PosRecScoreVAL': [] ,'PosFScore': [],
           'PosFScoreVAL': [],'NegPrecScore': [], 'NegPrecScoreVAL': [] ,'NegRecScore': [], 'NegRecScoreVAL': [],
           'NegFScore': [], 'NegFScoreVAL': [], 'TNPercentage': [], 'TNPercentageVAL': [],'TPPercentage': [], 
           'TPPercentageVAL': [],'FNPercentage': [], 'FNPercentageVAL': [], 'FPPercentage': [], 'FPPercentageVAL': []}
    
    # Deepcopy the classifierList
    models = deepcopy(classifierList)
    
    # Test each models in the list to verify 
    for i in range(len(classifierList)):
        try:
            model = classifierList[i]
            model.fit(X_train, y_train)
        except Exception as e:
            print("==============================================================")
            print(f"I wasn't able to score with the model: {classifications_list[i]}")
            print(f"This was the error I've received from my master:\n\n{e}.")
            print("\nI didn't let it faze me though, for now I've skipped this model.")
            print("==============================================================\n")
            models.remove(classifierList[i]) # Remove invalid models from list
    
    # Loop through all models
    for classifier in range(len(models)):
        # Destroy parenthesis and anything within
        modelName = re.sub(r"\([^()]*\)", '', str(models[classifier]))
        # Performance
        model = models[classifier]
        model.fit(X_train, y_train)          
        pred = model.predict(X_test)
        pred1 = model.predict(X_validate)
        # Results
        acc_score = accuracy_score(y_test, pred)
        acc_score1 = balanced_accuracy_score(y_validate, pred1) 
        noOfCorrect = accuracy_score(y_test, pred, normalize = False)
        noOfCorrect1 = balanced_accuracy_score(y_validate, pred1, adjusted = True) 
        total = noOfCorrect/acc_score
        total1 = noOfCorrect1/acc_score1
        Confusing = confusion_matrix(y_test, pred)
        madConfusing1 = confusion_matrix(y_validate, pred1)

        dpps = Confusing[1][1] / (Confusing[1][1] + Confusing[0][1]) # pos prec score
        dpps1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[0][1])
        dprs = Confusing[1][1] / (Confusing[1][1] + Confusing[1][0]) # pos rec score
        dprs1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[1][0])
        dpfs = 2 * (dpps * dprs) / (dpps + dprs) # pos f1 score
        dpfs1 = 2 * (dpps1 * dprs1) / (dpps1 + dprs1) # pos f1 score
        dnps = Confusing[0][0] / (Confusing[0][0] + Confusing[1][0]) # neg prec score
        dnps1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[1][0])
        dnrs = Confusing[0][0] / (Confusing[0][0] + Confusing[0][1]) # neg rec score
        dnrs1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[0][1])
        dnfs = 2 * (dnps * dnrs) / (dnps + dnrs) # neg f1 score
        dnfs1 = 2 * (dnps1 * dnrs1) / (dnps1 + dnrs1) 


        # Save everything
        dic['ModelName'].append(modelName)
        dic['AccuracyScore'].append(acc_score)
        dic['AccuracyScoreVAL'].append(acc_score1)
        dic['CorrectPredictionsCount'].append(noOfCorrect)
        dic['CorrectPredictionsCountVAL'].append(noOfCorrect1)
        dic['Total'].append(total)
        dic['TotalVAL'].append(total1)
        dic['PosPrecScore'].append(dpps)
        dic['PosPrecScoreVAL'].append(dpps1)
        dic['PosRecScore'].append(dprs)
        dic['PosRecScoreVAL'].append(dprs1)
        dic['PosFScore'].append(dpfs)
        dic['PosFScoreVAL'].append(dpfs1)
        dic['NegPrecScore'].append(dnps)
        dic['NegPrecScoreVAL'].append(dnps1)
        dic['NegRecScore'].append(dnrs)
        dic['NegRecScoreVAL'].append(dnrs1)
        dic['NegFScore'].append(dnfs)
        dic['NegFScoreVAL'].append(dnfs1)
        dic['TNPercentage'].append(Confusing[0][0]/total*100)
        dic['TNPercentageVAL'].append(madConfusing1[0][0]/total*100)
        dic['TPPercentage'].append(Confusing[1][1]/total*100)
        dic['TPPercentageVAL'].append(madConfusing1[1][1]/total*100)
        dic['FNPercentage'].append(Confusing[1][0]/total*100)
        dic['FNPercentageVAL'].append(madConfusing1[1][0]/total*100)
        dic['FPPercentage'].append(Confusing[0][1]/total*100)
        dic['FPPercentageVAL'].append(madConfusing1[0][1]/total*100)
        
    return pd.DataFrame.from_dict(dic)