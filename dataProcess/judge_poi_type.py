import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import os
from statics_get_poi import add_label


def get_time(row):
    return datetime.datetime.strptime(row['start_stay'], '%Y-%m-%d %H:%M:%S')


def get_duration_seconds(row):
    c = row['duration']
    c = c.split(' ')
    t = int(c[0]) * 24 * 3600
    tmp = c[-1].split(':')
    t += int(tmp[0]) * 3600
    t += int(tmp[1]) * 60
    t += int(float(tmp[2]))
    return t


def judge_type(value):
    if value['汽车服务_加油站_80'] > 0 and value['duration_seconds'] < 40 * 60:
        poi_type = 'gas'
    elif value['duration_seconds'] < 40 * 60 and ('18:30:00' < value['start_stay'][-8:] < '19:30:00' or
                                                  '06:30:00' < value['start_stay'][-8:] < '08:30:00' or
                                                  '11:30:00' < value['start_stay'][-8:] < '12:30:00'):
        poi_type = 'restaurant'
    else:
        poi_type = 'rest_area'
    return poi_type

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    cluster_id_poi = pd.read_csv('../all_data/sp_poi-11-12-2.csv', usecols=['cluster_id', '汽车服务_加油站_80'])

    sp_data_file = '../all_data/cluster_sp_11-12-2.csv'
    sp_data = pd.read_csv(sp_data_file)
    # 查看一共有多少停留点
    print(len(sp_data))
    sp_data = add_label(sp_data)
    sp_data = pd.merge(sp_data, cluster_id_poi, on='cluster_id', how='left')

    sp_data['new_start_stay'] = sp_data.apply(lambda x: get_time(x), axis=1)
    sp_data['duration_seconds'] = sp_data.apply(lambda x: get_duration_seconds(x), axis=1)
    sp_data['poi_type'] = sp_data.apply(lambda x: judge_type(x), axis=1)

    save_data = sp_data[['plan_no', 'dri_id', 'longitude', 'latitude',
                         'start_stay', 'duration', 'dist', 'poi_type']]
    save_data.to_csv('../all_data/without_merge_sp_poi-11-12-2.csv', index=False)