import pandas as pd
import os

pd.set_option('display.float_format', '{:.2f}'.format)
DATASET_PATH = '../datasets/wisdm-dataset/raw/'

def create_labels(df, user):
    """
    Function to create appropriate labels.csv file
    :param df: pd DataFrame
    :param user: user number
    :return:
    """

    # initialize lists and add first variables
    activity_list = []
    start_list = []
    end_list = []

    activity_list.append(df.loc[0, 'label'])
    start_list.append(df.loc[0, 'new_timestamps'])

    # loop over dataframe rows
    for index in range(1, max(df.index)-1):

        # check if label switches
        if df.loc[index, 'label'] != df.loc[index+1, 'label']:

            # if so, add to lists
            end_list.append(df.loc[index, 'new_timestamps'])
            activity_list.append(df.loc[index+1, 'label'])
            start_list.append(df.loc[index+1, 'new_timestamps'])

    # add last timestamp as well
    end_list.append(df.loc[max(df.index), 'new_timestamps'])

    # create dataframe and save it
    df = pd.DataFrame()
    df['label'] = pd.Series(activity_list)
    df['timestamps_start'] = pd.Series(start_list)
    df['timestamps_end'] = pd.Series(end_list)

    df.to_csv(f'{DATASET_PATH}users/{user}/labels.csv')

# loop over users
for user in range(1600,1611):
    if not os.path.exists(f'{DATASET_PATH}users/{user}/labels.csv'):
        df = pd.read_csv(f'{DATASET_PATH}users/{user}/accel_phone.txt', index_col=0)
        create_labels(df, user)