import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_1994=pd.DataFrame()
df_1998 = pd.DataFrame()
df_2002 = pd.DataFrame()
df_2006 = pd.DataFrame()
df_2010 = pd.DataFrame()
df_2014 = pd.DataFrame()
df_2018 = pd.DataFrame()
training_data_all=pd.DataFrame()

years=[1994,1998,2002,2006,2010,2014,2018]
# This module verifies the actual results for all the matches for which predictions were provided in the first module
# The actual results are obtained from the results.csv whereas the predictions for each WC are stored in
# Pred_Act_****.csv
for year_current in years:
    results = pd.read_csv('datasets/results.csv')
    if year_current==1994:
        pred = pd.read_csv('datasets/Pred_Act_1994.csv')
    elif year_current==1998:
        pred = pd.read_csv('datasets/Pred_Act_1998.csv')
    elif year_current==2002:
        pred=pd.read_csv('datasets/Pred_Act_2002.csv')
    elif year_current==2006:
        pred = pd.read_csv('datasets/Pred_Act_2006.csv')
    elif year_current==2010:
        pred = pd.read_csv('datasets/Pred_Act_2010.csv')
    elif year_current==2014:
        pred = pd.read_csv('datasets/Pred_Act_2014.csv')
    else:
        pred = pd.read_csv('datasets/Pred_Act_2018.csv')
    winner = []
    for i in range(len(results['home_team'])):
        if results['home_score'][i] > results['away_score'][i]:
            winner.append(results['home_team'][i])
        elif results['home_score'][i] < results['away_score'][i]:
            winner.append(results['away_team'][i])
        else:
            winner.append('Tie')

    year = []
    for row in results['date']:
        year.append(int(row[:4]))
    results['match_year'] = year

    # Filtering out the required results from results.csv
    results['winning_team'] = winner
    results=results[results.tournament=='FIFA World Cup']
    #if year_current==2002:
    results = results[results.match_year > year_current-1]
    results = results[results.match_year < year_current+1]

    # Ensure the names of the countries in the prediction and results file are the same.
    results.loc[results.home_team=='USA','home_team']='United States'
    results.loc[results.away_team=='USA','away_team']='United States'
    results.loc[results.winning_team=='USA','winning_team']='United States'

    pred.loc[pred.Team1=='USA','Team1']='United States'
    pred.loc[pred.Team2=='USA','Team2']='United States'
    pred.loc[pred.Predicted_Winner=='USA','Predicted_Winner']='United States'

    results.loc[results.home_team=='Cote Divoire','home_team']='Ivory Coast'
    results.loc[results.away_team=='Cote Divoire','away_team']='Ivory Coast'
    results.loc[results.winning_team=='Cote Divoire','winning_team']='Ivory Coast'

    pred.loc[pred.Team1=='Cote Divoire','Team1']='Ivory Coast'
    pred.loc[pred.Team2=='Cote Divoire','Team2']='Ivory Coast'
    pred.loc[pred.Predicted_Winner=='Cote Divoire','Predicted_Winner']='Ivory Coast'

    results.loc[results.home_team=='Korea Republic','home_team']='South Korea'
    results.loc[results.away_team=='Korea Republic','away_team']='South Korea'
    results.loc[results.winning_team=='Korea Republic','winning_team']='South Korea'

    pred.loc[pred.Team1=='Korea Republic','Team1']='South Korea'
    pred.loc[pred.Team2=='Korea Republic','Team2']='South Korea'
    pred.loc[pred.Predicted_Winner=='Korea Republic','Predicted_Winner']='South Korea'

    results.loc[results.home_team=='Ireland','home_team']='Republic of Ireland'
    results.loc[results.away_team=='Ireland','away_team']='Republic of Ireland'
    results.loc[results.winning_team=='Ireland','winning_team']='Republic of Ireland'

    results.loc[results.home_team=='Korea DPR','home_team']='North Korea'
    results.loc[results.away_team=='Korea DPR','away_team']='North Korea'
    results.loc[results.winning_team=='Korea DPR','winning_team']='North Korea'

    pred.loc[pred.Team1=='Korea DPR','Team1']='North Korea'
    pred.loc[pred.Team2=='Korea DPR','Team2']='North Korea'
    pred.loc[pred.Predicted_Winner=='Korea DPR','Predicted_Winner']='North Korea'

    pred.loc[pred.Team1=='China PR','Team1']='China'
    pred.loc[pred.Team2=='China PR','Team2']='China'
    pred.loc[pred.Predicted_Winner=='China PR','Predicted_Winner']='China'

    pred.loc[pred.Team1=='Yugoslavia','Team1']='Serbia'
    pred.loc[pred.Team2=='Yugoslavia','Team2']='Serbia'
    pred.loc[pred.Predicted_Winner=='Yugoslavia','Predicted_Winner']='Serbia'

    #print(results)
    pred['Actual Winner']=np.nan
    #print(pred)
    for index,row in pred.iterrows():
        #print(row)
        team1=row['Team1']
        team2=row.Team2
        for index2,row2 in results.iterrows():
            if row2.home_team ==team1:

                if row2.away_team==team2:
                    pred.iloc[index,4]=row2.winning_team
            if row2.away_team==team1:
                if row2.home_team==team2:
                    pred.iloc[index, 4] = row2.winning_team

    print(pred)
    if year_current==1994:
        df_1994=pred
        pred.to_csv('datasets/Pred_Act_1994_1.csv')
    elif year_current==1998:
        df_1998 = pred
        pred.to_csv('datasets/Pred_Act_1998_1.csv')
    elif year_current==2002:
        df_2002 = pred
        pred.to_csv('datasets/Pred_Act_2002_1.csv')
    elif year_current==2006:
        df_2006 = pred
        pred.to_csv('datasets/Pred_Act_2006_1.csv')
    elif year_current==2010:
        df_2010 = pred
        pred.to_csv('datasets/Pred_Act_2010_1.csv')
    elif year_current==2014:
        df_2014 = pred
        pred.to_csv('datasets/Pred_Act_2014_1.csv')
    else:
        df_2018 = pred
        pred.to_csv('datasets/Pred_Act_2018_1.csv')
training_data_all=pd.concat((df_1994,df_1998,df_2002,df_2006,df_2010,df_2014,df_2018),ignore_index=True)
training_data_all.to_csv('datasets/Pred_Act_All_3.csv',index = False)