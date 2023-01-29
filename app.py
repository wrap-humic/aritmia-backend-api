from flask import Flask, render_template, url_for, request, abort
from flask_mqtt import Mqtt
from flask_bcrypt import Bcrypt


import pandas as pd
import numpy as np
from scipy import stats
import pywt
import neurokit2 as nk
import math
import pickle
import data_dummy
from helper.response_helper import response_helper
from helper.upload_file_helper import handle_upload

from models import db
from models.User import User
from models.Pasien import Pasien
from models.Record import Record


app = Flask(__name__)

# Constant
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

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

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/wrap'
db.init_app(app)

# bcrypt
bcrypt = Bcrypt(app)


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

# Register


@app.route('/api/register', methods=['POST'])
def register():
    if (request.method == 'POST'):
        # get request data
        email = request.form['email']
        password = request.form['password']
        nama_lengkap = request.form['nama_lengkap']

        # Hash the password using bcrypt
        hashed_password = bcrypt.generate_password_hash(
            password).decode('utf-8')

        # Check if email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return response_helper(400, "Email sudah terpakai", None)

        # create object user
        new_user = User(email=email, password=hashed_password,
                        nama_lengkap=nama_lengkap)

        # validate,add user to the database, and show response
        try:
            db.session.add(new_user)
            db.session.commit()
            return response_helper(200, "Berhasil menambahkan data user", new_user.to_dict())
        except Exception as e:
            return response_helper(400, str(e), None)
    else:
        abort(404)


@app.route('/api/login', methods=['POST'])  # Login
def login():
    if (request.method == 'POST'):
        # get request data
        email = request.form['email']
        password = request.form['password']

        # Check if email and password already exists in the database
        try:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and bcrypt.check_password_hash(existing_user.password, password):
                user = User(email=email, password=None,
                            nama_lengkap=existing_user.nama_lengkap)
                return response_helper(200, "Berhasil Login", user.to_dict())
            else:
                return response_helper(400, "Username / Password salah", existing_user.model_to_dict())
        except Exception as e:
            return response_helper(400, str(e), None)
    else:
        abort(404)


@app.route('/api/patient', methods=['POST', 'GET'])  # patient route
def patient():
    # add patient
    if request.method == 'POST':
        # get request data
        nama = request.form['nama']
        jenis_kelamin = request.form['jenis_kelamin']
        umur = request.form['umur']
        tanggal_lahir = request.form['tanggal_lahir']
        kondisi_kesehatan = request.form['kondisi_kesehatan']

        patient = Pasien(nama=nama, jenis_kelamin=jenis_kelamin, umur=umur,
                         tanggal_lahir=tanggal_lahir, kondisi_kesehatan=kondisi_kesehatan)
        # validate,add patient to the database, and show response
        try:
            db.session.add(patient)
            db.session.commit()
            return response_helper(200, "Berhasil menambahkan data pasien", patient.to_dict())
        except Exception as e:
            return response_helper(400, str(e), None)
    # get all patient or get one patient only
    if request.method == 'GET':
        # get all patient
        try:
            patients = Pasien.query.all()
            # get of every patient object
            patients_list = [patient.to_dict() for patient in patients]
            return response_helper(200, "Berhasil mendapatkan data list pasien", patients_list)
        except Exception as e:
            return response_helper(400, str(e), None)


@app.route('/api/patient/<int:id>', methods=['GET'])  # patient by id
def patient_by_id(id=None):
    if request.method == 'GET':
        if id is None:
            response_helper(400, "Pasien tidak ditemukan", None)
        else:
            # filter patient by id
            try:
                patient = Pasien.query.filter_by(id=id).first()
                if patient is None:
                    response_helper(400, "Pasien tidak ditemukan", None)
                else:
                    return response_helper(200, "Berhasil mendapatkan data pasien", patient.to_dict())
            except Exception as e:
                return response_helper(400, str(e), None)


@app.route('/api/upload_record', methods=['POST'])  # upload excell files
def upload():
    if request.method == 'POST':
        record_id_owner = request.form['record_id_owner']
        record_owner = request.form['record_owner']
        record_date = request.form['record_date']

        # add file to directory
        is_success, file_name = handle_upload(
            record_id_owner, record_owner, record_date, request, ALLOWED_EXTENSIONS)

        if (is_success):
            # if success add file to directory add record data to database
            record = Record(id_pasien=record_id_owner,
                            path='data_user/'+file_name, tanggal_record=record_date)
            try:
                db.session.add(record)
                db.session.commit()
                return response_helper(200, 'Berhasil menambahkan file dan record', record.to_dict())
            except Exception as e:
                return response_helper(400, str(e), None)
        else:
            return response_helper(400, file_name, None)


# get record statistics
@app.route('/api/record_statistic/<int:id_pasien>', methods=['GET'])
def record_statistic(id_pasien: None):
    if request.method == 'GET':
        if id is None:
            response_helper(400, "Pasien tidak ditemukan", None)
        else:
            records = Record.query.filter_by(id_pasien=id_pasien)
            if records is None:
                response_helper(400, "Pasien tidak ditemukan", None)
            else:
                record_list = [record.to_dict() for record in records]
                record_list_stat = []
                for record in record_list:
                    df = pd.read_csv(record['path'])
                    pred = classification(df)
                    stat = {'Normal': pred.count('Normal'), 'Left bundle branch block beat': pred.count('Left bundle branch block beat'),
                            'Right bundle branch block beat': pred.count('Right bundle branch block beat'),
                            'Atrial premature beat': pred.count('Atrial premature beat'),
                            'Premature ventricular contraction': pred.count('Premature ventricular contraction')}

                    record_stat = {**record, **stat}
                    record_list_stat.append(record_stat)

                return response_helper(200, 'Berhasil get list record statistik', record_list_stat)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
