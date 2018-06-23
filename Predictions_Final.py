import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This module reads the output of the previous module which contains the predicted and the actual WC results starting from 1998 till 2018 (Initial few matches)
# Providing the actual and predicted results will act as a reinforcement for the system. Its a way to learn from the previous prediction mistakes which the model did

GroupA=['Russia','Uruguay','Egypt','Saudi Arabia']
GroupB=['Spain','Portugal','Iran','Morocco']
GroupC=['France','Denmark','Australia','Peru']
GroupD=['Croatia','Iceland','Argentina','Nigeria']
GroupE=['Brazil','Serbia','Switzerland','Costa Rica']
GroupF=['Sweden','Mexico','Germany','South Korea']
GroupG=['Belgium','England','Tunisia','Panama']
GroupH=['Japan','Senegal','Poland','Colombia']
#

pred = pd.read_csv('datasets/Pred_Act_All_3.csv') # Output of module 2
pred=pred.drop(['Year'],axis=1)
pred_train=pred[pred['Actual Winner'].notnull()]
pred_test=pred[pred['Actual Winner'].isnull()]
pred_test=pred_test.reset_index(drop=True)
pred_train['Pred_Accuracy'] = np.where((pred_train['Predicted_Winner'] == pred_train['Actual Winner'])
                     , 'Correct', 'Incorrect') # Not used. Prediction accuracy for previous WC is ~50% only. 
#pred_train.loc[pred_train['Predicted_Winner'] == pred_train['Actual Winner'],'Pred_Accuracy']='Correct'
#pred_train.loc[pred_train['Predicted_Winner'] != pred_train['Actual Winner'],'Pred_Accuracy']='Incorrect'
#print(pred_train.head())
#correct_count=pred_train[pred_train.Pred_Accuracy=='Correct'].count()
#incorrect_count=pred_train[pred_train.Pred_Accuracy=='Incorrect'].count()
#print(correct_count)

# Renaming the column Actual Winner to Actual_Winner
actual_winner=pred_train['Actual Winner'].values.tolist() 
pred_train['Actual_Winner']=actual_winner
pred_train=pred_train.drop(['Actual Winner'],axis=1)
#print(pred_train)

model_input = pd.get_dummies(pred_train, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(model_input.columns)
pred_train.loc[pred_train.Actual_Winner == pred_train.Team1, 'Actual_Winner']= 1
pred_train.loc[pred_train.Actual_Winner == 'Tie', 'Actual_Winner']= 0
pred_train.loc[pred_train.Actual_Winner == pred_train.Team2, 'Actual_Winner']= 2

feature_cols=['Team1','Team2','Predicted_Winner']
X=model_input.drop(['Actual_Winner'],axis=1)
X=X.drop(['Pred_Accuracy'],axis=1)
y=pred_train['Actual_Winner']
y=y.astype('int')

#print(X)
# Trying Log Reg on different values of C
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
C=[0.0001,0.001,0.01,0.1,1,1.5,2,2.5,3,3.5,4,4.5,100]
for i in C:
    logreg=LogisticRegression(C=i)
    logreg.fit(X_train, y_train)
    score = logreg.score(X_train, y_train)
    score2 = logreg.score(X_cv, y_cv)
    print(i)
    print(score)
    print(score2)
    print("")
#pred_test['Pred_Accuracy']='Incorrect'
    logreg=LogisticRegression(C=2.5)
    logreg.fit(X_train, y_train)
bkp_test_input=pred_test
test_input = pd.get_dummies(pred_test, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(test_input.columns)
print(pred_test.shape[0])

missing_cols = set(model_input.columns ) - set(test_input.columns )
for c in missing_cols:
    test_input[c] = 0
test_input = test_input[X.columns]
print(len(X.columns))
print(len(test_input.columns))
#test_input=test_input.drop(['Actual_Winner'],axis=1)
#train2,test2 = train.align(test, join='outer', axis=1, fill_value=0)
predictions = logreg.predict(test_input) # Get the predictions from the model which gives highest CV score. (For C=2.5)

#print(set(test_input.columns ) - set(X.columns ))
#print(set(test_input.columns ) - set(model_input.columns ))
#print(pred_test)
for i in range(pred_test.shape[0]):
    print(pred_test.iloc[i, 0] + " and " + pred_test.iloc[i, 1])
    if predictions[i] == 2:
        print("Winner: " + pred_test.iloc[i, 1])
    elif predictions[i] == 0:
        print("Tie")
    elif predictions[i] == 1:
        print("Winner: " + pred_test.iloc[i, 0])
    #print('Probability of ' + bkp_test_input.iloc[i, 0] + ' winning: ',
         # '%.3f' % (logreg.predict_proba(pred_test)[i][1]))
    #print('Probability of Tie: ', '%.3f' % (logreg.predict_proba(pred_test)[i][1]))
    #print('Probability of ' + pred_test.iloc[i, 1] + ' winning: ',
       #   '%.3f' % (logreg.predict_proba(pred_test)[i][2]))
    #print("")
	
# Predictions from Random Forest. Yeilds poor results compared to Log regression 
print ("Now with Random forest")
from sklearn.ensemble import RandomForestClassifier
# Having high value of estimators to ensure better robustness of the model. 
# Other parameters of the random forest did not have a major influence of the scores. 

clf = RandomForestClassifier(n_estimators =5000,max_features = "sqrt",oob_score = True)
clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_cv,y_cv))
predictions = clf.predict(test_input)
print("Now for Group 16")
# Manually calculating the scores and identifying the teams who move to KO stage.
# [Team1, Team2, Predicted_Winner of the match]
# Predicted winner of the match is obtained by running these matches on the module 1 (Initial_Predictions.py)
group_16=[['Serbia','Germany','Germany'],['England','Japan','England'],['Portugal','Russia','Portugal'],
          ['Denmark','Nigeria','Denmark'],['Mexico','Brazil','Brazil'],['Belgium','Senegal','Belgium'],['Spain','Uruguay','Spain'],
          ['France','Croatia','France']]
X_group_16=pd.DataFrame(group_16,columns=['Team1','Team2','Predicted_Winner'])
X_group_16_test = pd.get_dummies(X_group_16, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(test_input.columns)
#print(pred_test.shape[0])

missing_cols = set(model_input.columns ) - set(X_group_16_test.columns )
for c in missing_cols:
    X_group_16_test[c] = 0
X_group_16_test = X_group_16_test[X.columns]

predictions = logreg.predict(X_group_16_test)

for i in range(X_group_16.shape[0]):
    print(X_group_16.iloc[i, 0] + " and " + X_group_16.iloc[i, 1])
    if predictions[i] == 2:
        print("Winner: " + X_group_16.iloc[i, 1])
    elif predictions[i] == 0:
        print("Tie")
    elif predictions[i] == 1:
        print("Winner: " + X_group_16.iloc[i, 0])

print("Now for Quarters")
# Teams mentioned below are the predictions obtained in previous step. (Line 122 -133)
group_8=[['Spain','Denmark','Spain'],['Brazil','Belgium','Brazil'],['Portugal','France','Portugal'],
          ['Germany','England','Germany']]
X_group_8=pd.DataFrame(group_8,columns=['Team1','Team2','Predicted_Winner'])
X_group_8_test = pd.get_dummies(X_group_8, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(test_input.columns)
#print(pred_test.shape[0])

missing_cols = set(model_input.columns ) - set(X_group_8_test.columns )
for c in missing_cols:
    X_group_8_test[c] = 0
X_group_8_test = X_group_8_test[X.columns]

predictions = logreg.predict(X_group_8_test)

for i in range(X_group_8.shape[0]):
    print(X_group_8.iloc[i, 0] + " and " + X_group_8.iloc[i, 1])
    if predictions[i] == 2:
        print("Winner: " + X_group_8.iloc[i, 1])
    elif predictions[i] == 0:
        print("Tie")
    elif predictions[i] == 1:
        print("Winner: " + X_group_8.iloc[i, 0])

print("Now for Semis")
group_4=[['Germany','France','Germany'],['Spain','Brazil','Brazil']]
X_group_4=pd.DataFrame(group_4,columns=['Team1','Team2','Predicted_Winner'])
X_group_4_test = pd.get_dummies(X_group_4, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(test_input.columns)
#print(pred_test.shape[0])

missing_cols = set(model_input.columns ) - set(X_group_4_test.columns )
for c in missing_cols:
    X_group_4_test[c] = 0
X_group_4_test = X_group_4_test[X.columns]

predictions = logreg.predict(X_group_4_test)

for i in range(X_group_4.shape[0]):
    print(X_group_4.iloc[i, 0] + " and " + X_group_4.iloc[i, 1])
    if predictions[i] == 2:
        print("Winner: " + X_group_4.iloc[i, 1])
    elif predictions[i] == 0:
        print("Tie")
    elif predictions[i] == 1:
        print("Winner: " + X_group_4.iloc[i, 0])

print("Now for Finals")
group_2=[['Brazil','France','Brazil']]
X_group_2=pd.DataFrame(group_2,columns=['Team1','Team2','Predicted_Winner'])
X_group_2_test = pd.get_dummies(X_group_2, prefix=['Team1', 'Team2','Predicted_Winner'],
                             columns=['Team1', 'Team2','Predicted_Winner'])
#print(test_input.columns)
#print(pred_test.shape[0])

missing_cols = set(model_input.columns ) - set(X_group_2_test.columns )
for c in missing_cols:
    X_group_2_test[c] = 0
X_group_2_test = X_group_2_test[X.columns]

predictions = logreg.predict(X_group_2_test)

for i in range(X_group_2.shape[0]):
    print(X_group_2.iloc[i, 0] + " and " + X_group_2.iloc[i, 1])
    if predictions[i] == 2:
        print("Winner: " + X_group_2.iloc[i, 1])
    elif predictions[i] == 0:
        print("Tie")
    elif predictions[i] == 1:
        print("Winner: " + X_group_2.iloc[i, 0])







