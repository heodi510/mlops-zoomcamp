import pickle
import sys
import pandas as pd
from pathlib import Path


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def load_artifacts(file):
    with open(file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def gen_result(df, dv, model, year, month):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predicted_duration']=y_pred 
    df_result=df.loc[:,['ride_id', 'predicted_duration']]
    return df_result

def export_result(df_result, year, month):
    # Build and create the directory
    output_folder = Path("./output")
    output_folder.mkdir(parents=True, exist_ok=True)  # parents=True is harmless here

    # Construct the full path in an object-oriented way
    output_file = output_folder / f"homework_preds_{year}_{month:02d}.parquet"

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def print_mean_pred(df_result):
    print(df_result['predicted_duration'].mean())



def ride_duration_prediction(
    year:int,
    month:int):
        
    dv, model=load_artifacts('model.bin')
    df = read_data(f'yellow_tripdata_{year}-{month:02d}.parquet')
    df_result=gen_result(df, dv, model, year, month)
    export_result(df_result, year, month)
    print_mean_pred(df_result)

def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    ride_duration_prediction(
        year=year,
        month=month
    )


if __name__ == '__main__':
    run()

