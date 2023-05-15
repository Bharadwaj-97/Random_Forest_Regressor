import pandas as pd

# Try to add Wickets from scores.csv


main_data = pd.read_csv("score.csv")
first_column = main_data.columns[0]
main_data = main_data.drop([first_column], axis=1)
main_data=main_data.drop(columns=["batsman","nonstriker_batsman","bowler","run_out"])


df1 = main_data.copy()

df1.over = (df1.over.map(str) + "." + df1.ball.map(str)).map(float)

df2 = df1.copy()


df2["final_total_runs"] = 0
final_total_runs = df2.groupby(["match_id", "innings"]).runs.sum().reset_index()

for i in range(len(final_total_runs)):
    temp = final_total_runs.loc[i]
    df2.loc[(df2.match_id == temp.match_id) & (df2.innings == temp.inning   s), "final_total_runs"] = temp.runs


# let's create new column for wickets with value 0
df2["wickets"] = 0

# This will calculate Runs, wickets, balls till particular over
# let's get unique id
id = df2.match_id.unique()

dict = {'Chennai Super Kings':1, 'Mumbai Indians' : 2, 'Rajasthan Royals' :3,'Kolkata Knight Riders' :4,'Sunrisers Hyderabad': 5,'Delhi Capitals':6,'Kings XI Punjab':7, 'Royal Challengers Bangalore' :8}


for i in id:
    for inning in range(1, 3):
        index = df2.loc[(df2.match_id == i) & (df2.innings == inning)].index
        for j in range(len(index)):
            current_index_df = df2.loc[(df2.index == index[j])]
            previous_index_df = df2.loc[(df2.index == index[j - 1])]
            if j == 0:
                df2.loc[df2.index == index[j], "wickets"] = 0 if pd.isna(current_index_df.matchWicket.values[0]) else 1
            else:
                df2.loc[df2.index == index[j], "wickets"] = 0 if pd.isna(current_index_df.matchWicket.values[0]) else 1 + \
                                                            0 if pd.isna(previous_index_df.matchWicket.values[0]) else 1 
            


# Last 5 overs data
df2.loc[df2.over < int(str(current_index_df.over.values[0]).split(".")[0]) - 5]

df2["last_5_over_wickets"] = 0
df2["last_5_over_runs"] = 0
df2["batteam"]=0
df2["bowlteam"]=0
df2["runs_tillnow"] =0
df2["city"]=""
df2["winner"] = ""

for i in id:
    for inning in range(1, 3):
        index = df2.loc[(df2.match_id == i) & (df2.innings == inning)].index
        for j in range(len(index)):
            current_index_df = df2.loc[(df2.index == index[j])]
            previous_index_df = df2.loc[(df2.index == index[j-1])]
            df2.loc[df2.index == index[j], "runs_tillnow"] = current_index_df.runs.values[0]+previous_index_df.runs_tillnow.values[0]
            df2.loc[df2.index == index[j], "batteam"] = dict[current_index_df.batting_team.values[0]]
            df2.loc[df2.index == index[j], "bowlteam"] = dict[current_index_df.bowling_team.values[0]]
            if current_index_df.over.values[0] <= 5:

                df2.loc[df2.index == index[j], "last_5_over_wickets"] = (df2.loc[(df2.match_id == i) & (df2.innings == inning) & (df2.over < current_index_df.over.values[0]), "wickets"]).sum()
                df2.loc[df2.index == index[j], "last_5_over_runs"] = (df2.loc[(df2.match_id == i) & (df2.innings == inning) & (df2.over < current_index_df.over.values[0]), "runs"]).sum()
            else:
                df2.loc[df2.index == index[j], "last_5_over_wickets"] = (df2.loc[(df2.match_id == i) & (df2.innings == inning) & (df2.over < current_index_df.over.values[0]) & (df2.over > int(str(current_index_df.over.values[0]).split(".")[0]) - 5), "wickets"]).sum()
                df2.loc[df2.index == index[j], "last_5_over_runs"] = (df2.loc[(df2.match_id == i) & (df2.innings == inning) & (df2.over < current_index_df.over.values[0]) & (df2.over > int(str(current_index_df.over.values[0]).split(".")[0]) - 5), "runs"]).sum()



matches_data = pd.read_csv("matches copy.csv")
matches_data.drop(columns=["date","player_of_match","venue","neutral_venue","team1","team2","toss_winner","toss_decision","result","result_margin","eliminator","method","umpire1","umpire2"], inplace=True)
matches_data.dropna(inplace=True)


for i in id:
    temp = matches_data.loc[matches_data.id == i]
    df2.loc[df2.match_id == i, "city"] = temp.city.values[0]
    df2.loc[df2.match_id == i, "winner"] = temp.winner.values[0]

df2.drop(columns=["matchWicket", "isBoundary"], inplace=True)

df2.to_csv("feature_engineering6.csv", index=False)