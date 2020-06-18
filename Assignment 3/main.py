import pandas as pd
import os
from datetime import datetime, timedelta

DATAFILES = '../datasets/wisdm-dataset/raw/'
PHONE_ACCEL = 'phone/accel/'
PHONE_GYRO = 'phone/gyro/'
WATCH_ACCEL = 'watch/accel/'
WATCH_GYRO = 'watch/gyro/'
FOLDERS = [PHONE_ACCEL, PHONE_GYRO, WATCH_ACCEL, WATCH_GYRO]

timestamp_origin = datetime.now()

for folder in FOLDERS:
    timestamp = timestamp_origin
    for file in os.listdir(f'{DATAFILES}{folder}'):
        with open(f'{DATAFILES}{folder}/{file}') as f:
            lines = f.readlines()
            for line in lines:
                instance = line.split(',')
                instance[2] = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                timestamp += timedelta(milliseconds=50)
                #limit to reading one row
                print(instance)
                break



        #Limit to reading one file
        break


