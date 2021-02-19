import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    tf_mini = pd.read_csv('https://drive.google.com/uc?export=download&id=1Zcd6o-sM0iP5nNZF1FPbOiVpsN6o0KVJ') # size more than 25MB, uploaded on cloud.
    web_app_scaling_df = pd.read_csv('web_app_scaling_df.csv')       # size less than 25MB, uploaded on GitHub repository.

    form_data = {'session_position' : request.form['session_position'],  # request.form.get('session_position'), #
                 'session_length' : request.form['session_length'],
                 'track_id' : request.form['track_id'],
                 'context_switch' : request.form['context_switch'],
                 'no_pause_before_play' : request.form['no_pause_before_play'],
                 'hist_user_behavior_n_seekfwd' : request.form['hist_user_behavior_n_seekfwd'],
                 'hist_user_behavior_n_seekback' : request.form['hist_user_behavior_n_seekback'],
                 'hist_user_behavior_is_shuffle' : request.form['hist_user_behavior_is_shuffle'],
                 'hour_of_day' : request.form['hour_of_day'],
                 'date' : request.form['date'],
                 'premium' : request.form['premium'],
                 'context_type' : request.form['context_type'],
                 'hist_user_behavior_reason_start' : request.form['hist_user_behavior_reason_start'],
                 'hist_user_behavior_reason_end' : request.form['hist_user_behavior_reason_end']         }
    
    form_df = pd.DataFrame([list(form_data.values())], columns=list(form_data.keys()))
    
    # Merging form_df & track_df into single dataframe.
    session_track_data = pd.merge(form_df, tf_mini, on='track_id', how='left')
    
    # Replacing boolean (True, False) by int32 (1, 0)
    session_track_data.replace(['Yes', 'No'], [1, 0], inplace=True)
    # encoding the mode
    session_track_data['mode'].replace({'major': 1, 'minor': 0 }, inplace=True)
    
    # chaning the date to weekday and droping the date column
    session_track_data["date"] = pd.to_datetime(session_track_data["date"])
    session_track_data['week_day'] = session_track_data["date"].dt.dayofweek
    session_track_data.drop("date", inplace=True, axis=1)
    
    session_track_data.replace(['playbtn', 'remote', 'trackerror', 'endplay', 'clickrow'], 'merged', inplace=True)
    
    # setting one hot encoding for categorical columns (Nominal Columns)
    One_Hot_Encoder = OneHotEncoder()

    context_type = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['context_type']]).toarray())
    context_type.columns = One_Hot_Encoder.get_feature_names(['context_type'])

    hist_user_behavior_reason_start = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['hist_user_behavior_reason_start']]).toarray())
    hist_user_behavior_reason_start.columns = One_Hot_Encoder.get_feature_names(['hub_reason_start']) # hub = hist_user_behavior

    hist_user_behavior_reason_end = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['hist_user_behavior_reason_end']]).toarray())
    hist_user_behavior_reason_end.columns = One_Hot_Encoder.get_feature_names(['hub_reason_end'])  # hub = hist_user_behavior

    # Concatenate dataframe --> session_track_data + context_type + hist_user_behavior_reason_start + hist_user_behavior_reason_end
    session_track_data = pd.concat([session_track_data, context_type, hist_user_behavior_reason_start, hist_user_behavior_reason_end], axis = 1)

    session_track_data.drop(["context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end", "track_id"], axis = 1, inplace = True)
    
    # drop all highly correlated variables.
    session_track_data.drop(['beat_strength', 'danceability', 'dyn_range_mean'], axis=1, inplace=True)

    web_app_scaling_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    web_app_scaling_df = web_app_scaling_df.append(session_track_data)
    web_app_scaling_df.replace([np.nan], [0], inplace=True)
    web_app_scaling_df.reset_index(drop = True, inplace = True)
    
    # Scaling
    scaler = StandardScaler()
    for col in web_app_scaling_df.columns:
        if (len(web_app_scaling_df[col].unique()) != 2) :
            web_app_scaling_df[col] = scaler.fit_transform(np.array(web_app_scaling_df[col]).reshape(-1, 1))
            
    prediction = model.predict(web_app_scaling_df.tail(1))[0]
    output = ['Not Skip', 'Skip']

    return render_template('index.html', prediction_text= 'Prediction: Probably user will "{}" this Music track.'.format(output[prediction]))

if __name__ == "__main__":
    app.run(debug=True)
    
