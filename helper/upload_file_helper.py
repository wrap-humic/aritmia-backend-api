import os
import pandas as pd


def handle_upload(record_id_owner, record_owner, record_date, request, allowed_extensions):
    if record_owner in (None, '') or not record_owner.strip():
        return (False, 'Record owner can\'t empty')
    if record_date in (None, '') or not record_date.strip():
        return (False, 'Record date can\'t empty')
    if 'file' not in request.files:
        return (False, 'No file found')
    file = request.files['file']
    if file.filename == '':
        return (False, 'No file selected')
    if not file or not allowed_file(file.filename, allowed_extensions):
        return (False, 'Invalid file')

    # read the excel file using pandas
    df = pd.read_excel(file)

    # specify the folder to save the csv file but make it first if its does not exist
    if not os.path.exists('data_user'):
        os.makedirs('data_user')

    folder = 'data_user/'
    filename = f"{record_owner}_{record_id_owner}_{record_date}.csv"
    filepath = folder + filename
    i = 1
    while os.path.exists(filepath):
        i += 1
        filename = f"{record_owner}_{record_id_owner}_{record_date}_{i}.csv"
        filepath = folder + filename
    # save the dataframe to a csv file
    df.to_csv(filepath, index=False)
    return (True, filename)


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
