import pandas as pd
import os

# set starting time and end time
start_time = 1565882115000000000
end_time = 1565885355000000000
measure_time = end_time-start_time

# loop over users, device and sensor
for user in range(1600,1611):
    for device in ['phone', 'watch']:
        for sensor in ['accel', 'gyro']:

            df: pd.DataFrame = pd.read_csv(f'../datasets/wisdm-dataset/raw/{device}/{sensor}/data_{user}_{sensor}_{device}.txt',
                             names=['subject', 'label', 'timestamps', 'x', 'y', 'z'])

            # sort dataframe chronologically
            df.sort_values(by='timestamps', axis=0, inplace=True)

            # reset index to finish sorting
            df.reset_index(inplace=True, drop=True)
            timesteps = max(df.index)

            # add timedelta (assuming that it's proportionate)
            df['new_timestamps'] = start_time+df.index*(measure_time/timesteps)

            # save file
            df.to_csv(f'../datasets/wisdm-dataset/raw/{device}/{sensor}/data_{user}_{sensor}_{device}.txt')


# sort files per user
if not os.path.exists('../datasets/wisdm-dataset/raw/users/'):
    os.mkdir('../datasets/wisdm-dataset/raw/users/')

# replace files to user directory
for user in range(1600,1611):
    if not os.path.exists(f'../datasets/wisdm-dataset/raw/users/{user}/'):
        os.mkdir(f'../datasets/wisdm-dataset/raw/users/{user}/')
        for device in ['phone', 'watch']:
            for sensor in ['accel', 'gyro']:
                os.replace(f'../datasets/wisdm-dataset/raw/{device}/{sensor}/data_{user}_{sensor}_{device}.txt',
                           f'../datasets/wisdm-dataset/raw/users/{user}/{sensor}_{device}.txt')

