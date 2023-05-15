import pandas as pd
import json
import urllib3
from time import sleep

http = urllib3.PoolManager()
columns_data = pd.DataFrame()
match_data = pd.DataFrame()
periods = ["1","2"]
pages = ["1"]
leagueId="8048"
eventId=""


dat_url = pd.read_csv("csv_file.csv")
eventId_gp= [id for id in dat_url["match_id"]]
teams1= [team1 for team1 in dat_url["team_1"]]
teams2= [team2 for team2 in dat_url["team_2"]]

len_url= len(eventId_gp)
print(len_url)

for count in range(len_url):
    eventId = str(eventId_gp[count])
    for period in periods:
        for page in pages:
            sleep(15)
            column_data = pd.DataFrame()
            match_dat= http.request('GET', 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId='+leagueId+'&eventId='+eventId+'&period=' +period+ '&page='+page+'&filter=full&liveTest=false')
                  
                  
            if(len(match_dat.data)<100):
                break
            data = json.loads(match_dat.data)
            df = pd.json_normalize(data['comments'])
            column_data["ball"]=df["ball"]
            column_data["over"]=df["over"]
            column_data["runs"]=df["runs"]
            column_data["isBoundary"] = df["isBoundary"]
            
            if(period=="1"):               
                column_data["innings"]=1
            else:
                column_data["innings"]=2

            for bat,bowl in zip(df["currentBatsmen"],df["currentBowlers"]):
                column_data["batsman"] =(bat[0]["name"])
                column_data["nonstriker_batsman"] = bat[1]["name"]
                column_data["bowler"]=(bowl[0]["name"])
            
            if(str(df["currentInning.team.name"]) is teams1[count]):
                column_data["batting_team"] = teams2[count]
                column_data["bowling_team"] = teams1[count]
            else:
                column_data["batting_team"] = teams1[count]
                column_data["bowling_team"] = teams2[count]
                
            if("matchWicket.text" in df.columns):
                column_data["matchWicket"] = df["matchWicket.text"]
                column_data["matchWicket"].fillna("NA",inplace=True)
                column_data["run_out"]= ["Yes" if "run out" in wicket_text else "No" for wicket_text in column_data["matchWicket"]]
            else:
                column_data["matchWicket"]="NA"
                column_data["run_out"]="No"
            column_data["match_id"]=eventId  
                    

            matches_data = pd.concat([matches_data,column_data])   

matches_data.to_csv("score.csv")