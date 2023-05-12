import pandas as pd
import pickle
import os
import utils
from objclass import TrajPoint, Traj, TrajAll
from tqdm import tqdm
from time import time
DIST_THRESHOLD = 1000
DIST_ORI_THRESHOLD = 2000
SPEED_THRESHOLD = 33.3

def data_packing(fileIndex):
    t0 = time()
    path = '../all_data'
    filename = 'traj'
    traj_data = pd.read_csv(os.path.join(path, filename + fileIndex + '_preprocess' + '.csv'))
    t1 = time()
    print(t1-t0, 's')
    traj_all = TrajAll()
    for plan_no, plan_no_traj_data in tqdm(traj_data.groupby('plan_no')):
        trajectory = Traj(plan_no)
        dri_id = plan_no_traj_data['dri_id'].values[0]
        trajectory.dri_id = dri_id
        point_list = []
        for index, traj in plan_no_traj_data.iterrows():
            point = TrajPoint(traj['plan_no'], traj['longitude'], traj['latitude'], traj['time'], traj['dist'])
            point_list.append(point)
        trajectory.traj_point_list = point_list
        traj_all.traj_dict[plan_no] = trajectory
    return traj_all

def calculate_distance(traj_all):
    # Calculate the distance traveled from the origin to the current point
    last_point = None
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
                traj_point_obj.dist = 0
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                curr_point.dist = last_point.dist + dist
                last_point = curr_point
    return traj_all
#     f = open('save_traj' + fileIndex + '.pkl', 'wb')
#     pickle.dump(save_traj, f)
#     f.close()

def data_save(fileIndex, save_traj):
    save_data = []
    for plan, traj in tqdm(save_traj.traj_dict.items()):
        plan_no = traj.plan_no
        dri_id = traj.dri_id
        traj_list = traj.traj_point_list
        for point in traj_list:
            longitude = point.longitude
            latitude = point.latitude
            timestamp = point.time
            dist = point.dist
            save_data.append([plan_no, dri_id, longitude, latitude, timestamp, dist])

    df_save = pd.DataFrame(data=save_data, columns=['plan_no', 'dri_id', 'longitude',   
                                                    'latitude', 'time', 'dist'])
    save_path = '../all_data'
    save_filename = 'traj' + fileIndex + '_preprocess.csv'
    df_save.to_csv(os.path.join(save_path, save_filename), index=False)
    
if __name__ == '__main__':
    traj_all = data_packing('11-12-2')
    traj_all_with_dist = calculate_distance(traj_all)
    data_save('11-12-2', traj_all_with_dist)