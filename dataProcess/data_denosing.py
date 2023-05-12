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

# Extract trajectory

def data_denosing(fileIndex):
    t0 = time()
    path = '../all_data'
    filename = 'traj'
    traj_data = pd.read_csv(os.path.join(path, filename + fileIndex + '.csv'))
    t1 = time()
    print(t1-t0, 's')
    traj_all = TrajAll()
    for plan_no, plan_no_traj_data in tqdm(traj_data.groupby('plan_no')):
        trajectory = Traj(plan_no)
        dri_id = plan_no_traj_data['driver_id'].values[0]
        trajectory.dri_id = dri_id
        point_list = []
        for index, traj in plan_no_traj_data.iterrows():
            point = TrajPoint(traj['plan_no'], traj['longitude'], traj['latitude'], traj['time'])
            point_list.append(point)
        trajectory.traj_point_list = point_list
        traj_all.traj_dict[plan_no] = trajectory
    
    # Trajectory denoising, remove drift trajectory points
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        last_point = None
        re_index = []
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                diff_time = (pd.to_datetime(curr_point.time) - pd.to_datetime(last_point.time)).total_seconds()
                if diff_time == 0:
                    speed = 0
                else:
                    speed = dist / diff_time
                if speed > SPEED_THRESHOLD:
                    re_index.append(index)
                else:
                    last_point = curr_point
        if len(re_index) != 0:
            traj_obj.traj_point_list = [traj_obj.traj_point_list[i] for i in range(len(traj_obj.traj_point_list))
                                        if i not in re_index]

    # Trajectory denoising, remove trajectory with large trajectory point spacing
    save_traj = []
    for plan_no, traj_obj in tqdm(traj_all.traj_dict.items()):
        flag = False
        last_point = None
        if len(traj_obj.traj_point_list) == 0:
            continue
        for index, traj_point_obj in enumerate(traj_obj.traj_point_list):
            if index == 0:
                curr_point = traj_point_obj
                last_point = curr_point
            else:
                curr_point = traj_point_obj
                dist = utils.haversine_distance(curr_point, last_point)
                if dist > DIST_THRESHOLD:
                    flag = True
                    break
                last_point = curr_point
        if not flag:
            save_traj.append(traj_obj)
    return save_traj

def data_save(fileIndex, save_traj):
    save_data = []
    for traj in tqdm(save_traj):
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
    save_traj = data_denosing('11-12-2')
    data_save('11-12-2', save_traj)