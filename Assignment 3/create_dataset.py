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

def create_labels(user):
    """

    :return:
    """
    labels = pd.read_csv(f'{DATAFILES}user_{user}/labels.csv')
    activity_list = []
    start_list = []
    end_list = []

    activity_list.append(labels.loc[0, 'transport'])
    start_list.append(labels.loc[0, 'timestamps'])


    for index in range(1, max(labels.index)-1):

        if labels.loc[index, 'transport'] != labels.loc[index+1, 'transport']:


            end_list.append(labels.loc[index, 'timestamps'])
            activity_list.append(labels.loc[index+1, 'transport'])
            start_list.append(labels.loc[index+1, 'timestamps'])


    end_list.append(labels.loc[max(labels.index), 'timestamps'])

    df = pd.DataFrame()
    df['transport'] = pd.Series(activity_list)
    df['timestamps_start'] = pd.Series(start_list)
    df['timestamps_end'] = pd.Series(end_list)

    df.to_csv(f'{DATAFILES}user_{user}/labels.csv')


for user in range(1,11):
    values[f'user_{user}'] = dict()

    labels_df = pd.DataFrame()

    for file in os.listdir(f'{DATAFILES}U{user}'):

        labels_file = dict()
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
        with open(f'{DATAFILES}U{user}/{file}', 'r') as fw:
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
                    labels_file[start_time+int(line[0])] = label

        labels_file_df = pd.DataFrame.from_dict(labels_file, orient='index')
        labels_df = labels_df.append(labels_file_df)

    labels_df.sort_index()
    labels_df.reset_index(inplace=True)
    labels_df.rename(columns={'index': 'timestamps',
                              0: 'transport'}, inplace=True)
    labels_df.to_csv(f'{DATAFILES}/user_{user}/labels.csv')
    create_labels(user)

    if not os.path.exists(f'{DATAFILES}user_{user}'):
        os.mkdir(f'{DATAFILES}user_{user}')
    out_file = f'{DATAFILES}user_{user}'

    for value in values[f'user_{user}'].keys():
        to_dataframe(user, value, out_file)

