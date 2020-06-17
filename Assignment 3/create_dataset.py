import pandas as pd
import os

DATAFILES = '../datasets/raw_data/'

def to_dataframe(user, column, out_file):
    """

    :param column:
    :return:
    """
    x = values[f'user_{user}'][column]
    x_df = pd.DataFrame.from_dict(x, orient='index')
    x_df['sensor'] = column
    x_df.sort_index()
    x_df.reset_index(inplace=True)
    x_df.rename(columns={'index': 'timestamps'}, inplace=True)
    x_df.to_csv(f'{out_file}/{column}.csv')

values = dict()
for user in range(1,11):
    values[f'user_{user}'] = dict()

    user_path = f'{DATAFILES}U{user}'

    for file in os.listdir(user_path):

        if 'Bus' in file:
            label = 'bus'
        elif 'Car' in file:
            label = 'car'
        elif 'Still' in file:
            label = 'still'
        elif 'Train' in file:
            label = 'train'
        elif 'Walking' in file:
            label = 'walking'

        start_time = int(file[-17:-4])
        with open(f'{user_path}/{file}', 'r') as fw:
            lines = fw.read().splitlines()

            for line in lines:
                line = line.split(sep=',')


                sensor_index = line[1].split(sep='.')
                if len(sensor_index) > 1:
                    sensor = sensor_index[2]
                else:
                    sensor = sensor_index[0]


                if len(line) == 5:
                    if sensor not in values[f'user_{user}'].keys():
                        values[f'user_{user}'][sensor] = dict()
                    values[f'user_{user}'][sensor][start_time+int(line[0])] = dict(x=line[2], y=line[3], z=line[4],
                                                                             transport=label)

    if not os.path.exists(user_path):
        os.mkdir(user_path)
    out_file = f'{DATAFILES}user_{user}'

    for value in values[f'user_{user}'].keys():
        to_dataframe(user, value, out_file)
