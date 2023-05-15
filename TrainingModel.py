import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import gradio as gr

# Reading the dataset
ipl = pd.read_csv('feature_engineering3.csv')

# Data Cleaning by removing the columns not necessary for this model
ipl = ipl.drop(['run_out', 'match_id', 'batting_team', 'bowling_team', 'winner', 'city'], axis=1)
ipl.rename(columns={'batsman': 'striker'}, inplace=True)

ipl_dict = {'Chennai Super Kings': 1, 'Mumbai Indians': 2, 'Rajasthan Royals': 3, 'Kolkata Knight Riders': 4,
            'Sunrisers Hyderabad': 5, 'Delhi Capitals': 6, 'Kings XI Punjab': 7, 'Royal Challengers Bangalore': 8}

# Replacing the Null values with .
str_cols = ipl.columns[ipl.dtypes == object]
ipl[str_cols] = ipl[str_cols].fillna('.')

# Encoding categories in non-numeric values like strings to Numeric values for training

Feature_list = []

# An object is created for each column and store it in feature list
for c in ipl.columns:
    if ipl.dtypes.dtypes == object:
        print(c, "->", ipl.dtypes.dtypes)
        Feature_list.append(c)

# Get unique values
a2 = ipl['striker'].unique()
a3 = ipl['bowler'].unique()
a4 = ipl['nonstriker_batsman'].unique()


# Each value will be should be converted to some numeric value for training
# It can be done Manually by creating a dictionary and assigning a value
# but we can use fit() of Label Encoder() of Scikit-learn library and store it in dictionary feature_dict

def labelEncoding(data):
    # Inputting dataset as Dataframe for pandas
    dataset = pd.DataFrame(ipl)
    feature_dict = {}

    for feature in dataset:
        if dataset[feature].dtype == object:
            le = preprocessing.LabelEncoder()
            fs = dataset[feature].unique()
            le.fit(fs)
            dataset[feature] = le.transform(dataset[feature])
            feature_dict[feature] = le

    return dataset


labelEncoding(ipl)

ip_dataset = ipl[['innings', 'batteam',
                  'bowlteam', 'striker', 'nonstriker_batsman',
                  'bowler', 'ball', 'over', 'runs', 'runs_tillnow', 'wickets', 'last_5_over_wickets',
                  'last_5_over_runs']]

b2 = ip_dataset['striker'].unique()
b3 = ip_dataset['bowler'].unique()
b4 = ip_dataset['nonstriker_batsman'].unique()
ipl.fillna(0, inplace=True)

features = {}

for i in range(len(a2)):
    features[a2[i]] = b2[i]
for i in range(len(a3)):
    features[a3[i]] = b3[i]
for i in range(len(a4)):
    features[a4[i]] = b4[i]

features

X = ipl[['over', 'batteam', 'bowlteam', 'last_5_over_runs', 'last_5_over_wickets', 'runs_tillnow']].values

Y = ipl['final_total'].values

# Splitting the dataset into the Training set and Test set
# We used SKLearn Train_test_split so that we can get a unbiased split of data set into Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the dataset

# n_estimators is the no.of tress in random forest, Controlling the tree from Overfitting
reg = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, max_depth=None, max_features='auto', max_samples=None,
                            min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100,
                            n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)

reg.fit(X_train, y_train)

# Testing the dataset on trained model
y_pred = reg.predict(X_test)
sample = pd.DataFrame(y_pred, columns=['Predict'])
sample['Actual'] = y_test

# Printing the sample 10 results for inspection
print(sample.head(20))

score = reg.score(X_test, y_test) * 100
print("R square value:", score)

css_code = 'body{background-image:url("https://www.deccanherald.com/sites/dh/files/styles/gallery_thumbnails/public/gallery_images/2022/03/25/Lead.png");background-size:cover;background-repeat:no-repeat; background-position: right center;}'


def testing(over, ball_in_the_over, batting_team, bowling_team, runs_in_last_5_overs, wickets_in_last_5_overs,
            current_score):
    # Testing with a custom input, Runs, Wickets, Overs, Striker, Non Striker
    over_here = str(over) + "." + str(ball_in_the_over);
    batting_team = ipl_dict.get(batting_team)
    bowling_team = ipl_dict.get(bowling_team)
    if bowling_team == batting_team:
        return "Batting Team and Bowling Team cannot be same"
    elif runs_in_last_5_overs > current_score:
        return "Runs scored in 5 overs cannot be more than the total runs scored."
    elif runs_in_last_5_overs > 150:
        return " Its impossible to score these runs in 5 overs"
    elif current_score > 300:
        return " Not possible to score these runs in T20 cricket"
    else:
        new_prediction = reg.predict(sc.transform(np.array(
            [[over_here, batting_team, bowling_team, runs_in_last_5_overs, wickets_in_last_5_overs, current_score]])))
        return (new_prediction.astype(int))


iface = gr.Interface(

    testing,
    [
        gr.inputs.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        gr.inputs.Dropdown([1, 2, 3, 4, 5, 6]),
        gr.inputs.Dropdown(['Chennai Super Kings', 'Delhi Capitals','Kings XI Punjab', 'Kolkata Knight Riders','Mumbai Indians', 'Rajasthan Royals',
                            'Royal Challengers Bangalore','Sunrisers Hyderabad' ]),
        gr.inputs.Dropdown(['Chennai Super Kings', 'Delhi Capitals','Kings XI Punjab', 'Kolkata Knight Riders','Mumbai Indians', 'Rajasthan Royals',
                            'Royal Challengers Bangalore','Sunrisers Hyderabad']),
        gr.inputs.Number(default=0, label=None, optional=False),
        gr.inputs.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        gr.inputs.Number(default=0, label=None, optional=False),
    ],
    gr.outputs.Textbox(type= "auto"),
    css=css_code,
    allow_screenshot=False,
    allow_flagging="never",

)

iface.launch()
