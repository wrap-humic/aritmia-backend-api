from flask import Flask, render_template, url_for, request
from flask_mqtt import Mqtt

import pandas as pd
import numpy as np
from scipy import stats
import pywt
import neurokit2 as nk
import math
import pickle
import data_dummy


app = Flask(__name__)

app.config['MQTT_BROKER_URL'] = 'broker.hivemq.com'
app.config['MQTT_BROKER_PORT'] = 1883
# Set this item when you need to verify username and password
app.config['MQTT_USERNAME'] = ''
# Set this item when you need to verify username and password
app.config['MQTT_PASSWORD'] = ''
app.config['MQTT_KEEPALIVE'] = 5  # Set KeepAlive time in seconds
# If your broker supports TLS, set it True
app.config['MQTT_TLS_ENABLED'] = False
topic = 'rhythm/ECG006/ecg'

mqtt_client = Mqtt(app)


@app.route('/')
def index():
    return "hyuguygu"

# @mqtt_client.on_connect()
# def handle_connect(client, userdata, flags, rc):
#    if rc == 0:
#        print('Connected successfully')
#        mqtt_client.subscribe(topic) # subscribe topic
#    else:
#        print('Bad connection. Code:', rc)

# @mqtt_client.on_message()
# def handle_mqtt_message(client, userdata, message):
#   # START: Preprocessing data signal
#   ecgSignal=[]
#   for i in range(5):
#     ecgSignalList = message.payload.decode()
#     ecgSignalList = ecgSignalList.split(":")
#     ecgSignalList.pop(0)
#     ecgSignalList = list(map(float, ecgSignalList))
#     for j in range(len(ecgSignalList)):
#       ecgSignal.append(ecgSignalList[j])

#   # END: Preprocessing data signal

# START: Machine Learning Model


def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec


def Feature_Extraction(data):
    rpeak = []
    vrpeak = []
    vp = []
    vq = []
    vs = []
    vt = []
    posq = []
    posis = []
    QRS = []
    RR = []

    Signal = np.array(data)
    Signal = denoise(Signal)
    Signal = stats.zscore(Signal)
    _, rpeak_temp = nk.ecg_peaks(Signal, sampling_rate=200)
    _, waves_peak = nk.ecg_delineate(
        Signal, rpeak_temp, sampling_rate=200, method="peak")
    for i in range(len(rpeak_temp['ECG_R_Peaks'])):
        vrpeak.append(Signal[rpeak_temp['ECG_R_Peaks'][i]])
        rpeak.append(rpeak_temp['ECG_R_Peaks'][i])
    for i in range(len(waves_peak['ECG_T_Peaks'])):
        if (math.isnan(waves_peak['ECG_P_Peaks'][i])):
            vp.append(np.nan)
        else:
            vp.append(Signal[waves_peak['ECG_P_Peaks'][i]])

        if (math.isnan(waves_peak['ECG_Q_Peaks'][i])):
            vq.append(np.nan)
        else:
            vq.append(Signal[waves_peak['ECG_Q_Peaks'][i]])

        if (math.isnan(waves_peak['ECG_S_Peaks'][i])):
            vs.append(np.nan)
        else:
            vs.append(Signal[waves_peak['ECG_S_Peaks'][i]])

        if (math.isnan(waves_peak['ECG_T_Peaks'][i])):
            vt.append(np.nan)
        else:
            vt.append(Signal[waves_peak['ECG_T_Peaks'][i]])
        posq.append(waves_peak['ECG_Q_Peaks'][i])
        posis.append(waves_peak['ECG_S_Peaks'][i])
    for i in range(len(vs)):
        qrs = posis[i]-posq[i]
        QRS.append(qrs)
    for i in range(len(rpeak)):
        if (i != len(rpeak)-1):
            rr = rpeak[i+1]-rpeak[i]
            RR.append(rr)
    df = pd.DataFrame({'ECG_R_Peaks': vrpeak,
                       'ECG_P_Peaks': vp,
                      'ECG_Q_Peaks': vq,
                       'ECG_S_Peaks': vs,
                       'ECG_T_Peaks': vt,
                       'QRS': QRS, })
    df.drop(df.tail(1).index, inplace=True)
    df['RR Interval'] = RR
    return df


def classification(features):
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    features['Prediksi'] = pickled_model.predict(features)
    hasil = np.array(features['Prediksi'])
    hasilcat = []
    for i in range(len(hasil)):
        if (hasil[i] == 'N'):
            hasilcat.append('Normal')
        elif (hasil[i] == 'L'):
            hasilcat.append('Left bundle branch block beat')
        elif (hasil[i] == 'R'):
            hasilcat.append('Right bundle branch block beat')
        elif (hasil[i] == 'A'):
            hasilcat.append('Atrial premature beat')
        else:
            hasilcat.append('Premature ventricular contraction')
    return hasilcat


@app.route('/AI', methods=['GET'])
def ensemblelearning():
    # data = data_dummy.data_dummy()
    # df = Feature_Extraction(data)
    df = pd.read_csv('DataArythmia Clean.csv')
    pred = classification(df)
    response = {"status": "ok", "data": pred, "message": "List of status"}
    return response
# END: Machine Learning Model


@app.route('/classify', methods=['POST'])
def classify():
    df = pd.DataFrame([{'ECG_R_Peaks': request.form['ecgrpeaks'],
                        'ECG_P_Peaks': request.form['ecgppeaks'],
                        'ECG_Q_Peaks': request.form['ecgqpeaks'],
                        'ECG_S_Peaks': request.form['ecgspeaks'],
                        'ECG_T_Peaks': request.form['ecgtpeaks'],
                        'QRS': request.form['qrs'],
                        'RR Interval': request.form['rrinterval'],
                        }])
    pred = classification(df)
    print(df)
    response = {"status": "ok", "data": pred[0], "message": "Classification"}
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
