import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The inital_predictions.py is a liftoff from https://github.com/brunoaks/FIFA-worldcup-2018-prediction
#  

# This module will give predictions for all World cup matches starting from 1994 without knowing the actual results.
wc = pd.read_csv('datasets/World Cup 2018 Dataset.csv')
#print(wc.columns)
results = pd.read_csv('datasets/results.csv')
#print(results.head())

# Results csv file has the results of all the footbacll matches that were played since 1900s
winner = []
for i in range(len(results['home_team'])):
    if results['home_score'][i] > results['away_score'][i]:
        winner.append(results['home_team'][i])
    elif results['home_score'][i] < results['away_score'][i]:
        winner.append(results['away_team'][i])
    else:
        winner.append('Tie')
results['winning_team'] = winner
# Adding new column for goal difference in matches
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])
#print(results.head())


df_1994=pd.DataFrame()
df_1998 = pd.DataFrame()
df_2002 = pd.DataFrame()
df_2006 = pd.DataFrame()
df_2010 = pd.DataFrame()
df_2014 = pd.DataFrame()
df_2018 = pd.DataFrame()
#Change 1 - Include teams only for that WC for each WC
years=[1994,1998,2002,2006,2010,2014,2018]
for year_current in years:
#year_current=2010
    if year_current==1994:
        wc_teams = ['Saudi Arabia', ' South Korea', 'Cameroon', 'Morocco',
                'Nigeria', 'Mexico', 'United States', 'Argentina',
                'Brazil', 'Bolivia', 'Colombia', 'Belgium',
                'Bulgaria', 'Germany', 'Greece', 'Italy',
                'Netherlands', 'Norway', 'Republic of Ireland', 'Romania',
                'Russia', 'Spain', 'Sweden', 'Switzerland']
    elif year_current==1998:
        wc_teams = ['Cameroon', 'Morocco', 'Nigeria', 'South Africa',
                    'Tunisia', 'Iran', 'Japan', 'South Korea',
                    'Saudi Arabia', 'Austria', 'Belgium', 'Bulgaria',
                    'Croatia', 'Denmark', 'England', 'France',
                    'Germany', 'Italy', 'Netherlands', 'Norway',
                    'Romania', 'Scotland', 'Spain', 'Yugoslavia', 'Jamaica'
            , 'Mexico', 'United States']
    elif year_current==2002:
        wc_teams = ['Saudi Arabia', ' South Korea', 'China', 'Japan', 'Slovenia'
                   'Nigeria', 'Cameroon', 'Senegal', 'South Africa','Turkey'
                    'Tunisia', 'Costa Rica', 'Mexico', 'United States',
                    'Argentina', 'Germany', 'Belgium', 'Italy', 'Croatia'
                    'Poland', 'Portugal', 'Denmark', 'Republic of Ireland',
                    'Russia''France', 'Spain', 'Sweden', 'England', 'Brazil', 'Ecuador', 'Paraguay', 'Uruguay']
    elif year_current==2006:
        wc_teams = ['Saudi Arabia', ' South Korea', 'Iran', 'Japan', 'Angola'
                    'Ghana', 'Ivory Coast', 'Togo', 'Trinidad and Tobago','Tunisia',
                    'Costa Rica', 'Mexico', 'United States', 'Australia'
                    'Argentina', 'Germany', 'Netherlands', 'Italy', 'Croatia','Ukraine'
                    'Poland', 'Portugal', 'Czech Republic', 'Serbia and Montenegro', 'Switzerland'
                    'France', 'Spain', 'Sweden', 'England','Brazil', 'Ecuador', 'Paraguay']
    elif year_current==2010:
        wc_teams = ['South Korea', 'North Korea', 'Japan', 'Algeria'
                    'Ghana', 'Ivory Coast', 'Cameroon', 'Nigeria', 'Australia'
                    'South Africa','Honduras', 'Mexico', 'United States', 'New Zealand'
                    'Argentina', 'Germany', 'Netherlands', 'Italy', 'Serbia','Denmark'
                    'Slovakia', 'Portugal', 'Greece', 'Chile', 'Switzerland'
                    'France', 'Spain', 'Slovenia', 'England', 'Brazil','Uruguay', 'Paraguay']
    elif year_current==2014:
        wc_teams = ['South Korea', 'Iran', 'Japan', 'Algeria','Ghana', 'Ivory Coast', 'Cameroon', 'Nigeria', 'Australia'
                    'Honduras', 'Mexico','United States', 'Costa Rica', 'Ecuador'
                    'Argentina', 'Germany', 'Netherlands', 'Italy', 'Bosnia and Herzegovina','Belgium'
                    'Russia', 'Portugal', 'Croatia', 'Chile', 'Switzerland'
                    'France', 'Spain', 'Greece', 'England', 'Brazil', 'Uruguay','Colombia']
    else:
        wc_teams = ['Australia', ' Iran', 'Japan', 'Korea Republic',
                    'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria',
                    'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
                    'Panama', 'Argentina', 'Brazil', 'Colombia',
                    'Peru', 'Uruguay', 'Belgium', 'Croatia',
                    'Denmark', 'England', 'France', 'Germany',
                    'Iceland', 'Poland', 'Portugal', 'Russia',
                    'Serbia', 'Spain', 'Sweden', 'Switzerland']

    # Filter the 'results' dataframe to show only teams in this years' world cup, from 1930 onwards.
    #We do not need the data of other teams which are not participating in the given WC

    df_teams_home = results[results['home_team'].isin(wc_teams)]
    df_teams_away = results[results['away_team'].isin(wc_teams)]
    df_teams = pd.concat((df_teams_home, df_teams_away))
    df_teams.drop_duplicates()
    #print(df_teams.count())
    year = []
    for row in df_teams['date']:
        year.append(int(row[:4]))
    df_teams['match_year'] = year

    # Change 2 - Restrict the data to year in which WC was held.
    # For predicting the results of 1994 WC, we would need the historical data only till 1994 and nothing after that.

    df_teams30 = df_teams[df_teams.match_year >= 1930]
    if year_current==1994:
        #print("Inside i==1994")
        df_teams30 = df_teams[df_teams.match_year < 1994]
    elif year_current==1998:
        df_teams30 = df_teams[df_teams.match_year < 1998]
    elif year_current==2002:
        df_teams30 = df_teams[df_teams.match_year < 2002]
    elif year_current==2006:
        df_teams30 = df_teams[df_teams.match_year < 2006]
    elif year_current==2010:
        df_teams30 = df_teams[df_teams.match_year < 2010]
    elif year_current==2014:
        df_teams30 = df_teams[df_teams.match_year < 2014]
    else:
        df_teams30=df_teams30
    #print(df_teams30.head())
    #Drop unwanted columns.
    df_teams30 = df_teams30.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], axis=1)
    #print(df_teams30.head(5))

    df_teams30 = df_teams30.reset_index(drop=True)
    df_teams30.loc[df_teams30.winning_team == df_teams30.home_team, 'winning_team']= 2
    df_teams30.loc[df_teams30.winning_team == 'Tie', 'winning_team']= 1
    df_teams30.loc[df_teams30.winning_team == df_teams30.away_team, 'winning_team']= 0

    #print(df_teams30.head())
    from sklearn.model_selection import train_test_split
    final = pd.get_dummies(df_teams30, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Separate X and y sets
    X = final.drop(['winning_team'], axis=1)
    y = final["winning_team"]
    y = y.astype('int')

    # Separate train and test sets
    #X_test is more of the cross validation dataset. THe real test data is the actual WC matches of the given year
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    score = logreg.score(X_train, y_train)
    score2 = logreg.score(X_test, y_test)

    print("Training set accuracy: ", '%.3f'%(score))
    print("Test set accuracy: ", '%.3f'%(score2))
    #sns.countplot(x='winning_team', data=df_teams30)
    #plt.show()
    # The current FIFA rankings of 2018 wont help to predict results for 1994 WC.
    # Get the FIFA rankings of 1994 if we are interested to get predictions for 1994 WC
    if year_current==1994:
        #print("Inside i==1994")
        ranking = pd.read_csv('datasets/fifa_rankings_1994.csv')
        fixtures = pd.read_csv('datasets/fixtures_1994.csv') #
    elif year_current==1998:
        ranking = pd.read_csv('datasets/fifa_rankings_1998.csv')
        fixtures = pd.read_csv('datasets/fixtures_1998.csv')
    elif year_current==2002:
        ranking = pd.read_csv('datasets/fifa_rankings_2002.csv')
        fixtures = pd.read_csv('datasets/fixtures_2002.csv')
    elif year_current==2006:
        ranking = pd.read_csv('datasets/fifa_rankings_2006.csv')
        fixtures = pd.read_csv('datasets/fixtures_2006.csv')
    elif year_current==2010:
        ranking = pd.read_csv('datasets/fifa_rankings_2010.csv')
        fixtures = pd.read_csv('datasets/fixtures_2010.csv')
    elif year_current==2014:
        ranking = pd.read_csv('datasets/fifa_rankings_2014.csv')
        fixtures = pd.read_csv('datasets/fixtures_2014.csv')
    else:
        #print("Inside i==2018")
        ranking = pd.read_csv('datasets/fifa_rankings.csv')
        fixtures = pd.read_csv('datasets/fixtures.csv')

    #print(ranking)
    # List for storing the group stage games
    pred_set = []
    # Create new columns with ranking position of each team
    temp=fixtures['Home Team'].map(ranking.set_index('Team')['Position'])
    #print(temp)
    fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
    fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))

    # We only need the group stage games, so we have to slice the dataset
    # For the previous WC, predict all the matches including the KO rounds.
    # For 2018, predict only the group stages
    if year_current==1994:
        fixtures = fixtures.iloc[:52, :]# Change - 4
    elif year_current==1998:
        fixtures = fixtures.iloc[:64, :]
    elif year_current==2002:
        fixtures = fixtures.iloc[:64, :]
    elif year_current==2006:
        fixtures = fixtures.iloc[:64, :]
    elif year_current==2010:
        fixtures = fixtures.iloc[:64, :]
    elif year_current==2014:
        fixtures = fixtures.iloc[:64, :]
    else: # For 2018
        fixtures = fixtures.iloc[:48, :]
    #print(fixtures.tail())

    for index, row in fixtures.iterrows():
        if row['first_position'] < row['second_position']:
            pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
        else:
            pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None})

    pred_set = pd.DataFrame(pred_set)
    #Actual test data set
    backup_pred_set = pred_set

    #print(pred_set.head())
    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols = set(final.columns) - set(pred_set.columns)
    for c in missing_cols:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    #print(pred_set.head())
    predictions = logreg.predict(pred_set)
    Year=[]
    Team1=[]
    Team2=[]
    Predicted_Winner=[]
    training_data=pd.DataFrame()
    training_data_all=pd.DataFrame()
    for i in range(fixtures.shape[0]):
        Year.append(year_current)
        Team1.append(backup_pred_set.iloc[i, 1])
        Team2.append(backup_pred_set.iloc[i, 0])
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
            Predicted_Winner.append(backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Tie")
            Predicted_Winner.append('Tie')
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
            Predicted_Winner.append(backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Tie: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1]))
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
        print("")
    training_data['Year']=Year
    training_data['Team1']=Team1
    training_data['Team2']=Team2
    training_data['Predicted_Winner']=Predicted_Winner
    print(training_data)
    #training_data_all.append(training_data)
    #Saving pred of each World cup individually and combining them all in one single file as well
    if year_current==1994:
        df_1994=training_data
        training_data.to_csv('datasets/Pred_Act_1994.csv',index=False)
    elif year_current==1998:
        df_1998=training_data
        training_data.to_csv('datasets/Pred_Act_1998.csv',index=False)
    elif year_current==2002:
        df_2002=training_data
        training_data.to_csv('datasets/Pred_Act_2002.csv',index=False)
    elif year_current==2006:
        df_2006=training_data
        training_data.to_csv('datasets/Pred_Act_2006.csv',index=False)
    elif year_current==2010:
        df_2010=training_data
        training_data.to_csv('datasets/Pred_Act_2010.csv',index=False)
    elif year_current==2014:
        df_2014=training_data
        training_data.to_csv('datasets/Pred_Act_2014.csv',index=False)
    else:
        df_2018=training_data
        training_data.to_csv('datasets/Pred_Act_2018.csv',index=False)
training_data_all=pd.concat((df_1994,df_1998,df_2002,df_2006,df_2010,df_2014,df_2018),ignore_index=True)
training_data_all.to_csv('datasets/Pred_Act_All_2.csv',index = False)

