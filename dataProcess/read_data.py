import pandas as pd
import os
import time
# import matplotlib.pyplot as plt
def extract_traj(file_name):
    # Extract selected trajectory
    file_path = '../all_data/'
    t1 = time.time()
    
    traj_reader = pd.read_csv(os.path.join(file_path, file_name), iterator=True, low_memory = False, chunksize = 100000,
                                                                                                          usecols = ['plan_no', 'truck_no', 'time', 
                                                                                                        'longitude', 'latitude', 'speed'])
    chunk_list = []
    for chunk in traj_reader:
        chunk_list.append(chunk) 
    traj_flow = pd.concat(chunk_list, ignore_index=True)
    t2 = time.time()
    print(t2-t1, 's')

    # Read waybill data, merge trajectory data and add driver_id field
    waybill_reader = pd.read_csv(os.path.join(file_path, 'waybill_data_name.csv'), iterator=True, low_memory = False, chunksize = 100000,
                                                                                                            usecols = ['plan_no', 'driver_id'])
    chunk_list = []
    for chunk in waybill_reader:
        chunk_list.append(chunk) 
    waybill_data = pd.concat(chunk_list, ignore_index=True)
    traj_flow = pd.merge(traj_flow, waybill_data[['plan_no', 'driver_id']], on='plan_no', how='left')
    traj_flow.info()

    # Delete trajectories with too many or too few trajectory points
    tmp = traj_flow['plan_no'].value_counts().tolist()
    plt.hist(tmp, bins=40, facecolor="blue", edgecolor="black", alpha=0.7, range=(240, 10000)
             )  # weights=np.ones(len(tmp)) / len(tmp)
    traj_flow_re_longshort = traj_flow.groupby('plan_no').filter(lambda x: 240 <= len(x) <= 10000)
    return traj_flow_re_longshort

if __name__ == '__main__':
    traj_flow_11_12_1 = extract_traj('trajctory11-12-1.csv')
    traj_flow_11_12_1.dropna(subset = ['driver_id'], axis = 0, inplace = True)
    traj_flow_11_12_1.to_csv('../all_data/traj11-12-1.csv', index=False)
    
    traj_flow_11_12_2 = extract_traj('trajctory11-12-2.csv')    
    traj_flow_11_12_2.dropna(subset = ['driver_id'], axis = 0, inplace = True)
    traj_flow_11_12_2.to_csv('../all_data/traj11-12-2.csv', index=False)
    
    traj_flow_1_2 = extract_traj('trajctory1-2.csv')
    traj_flow_1_2.dropna(subset = ['driver_id'], axis = 0, inplace = True)
    traj_flow_1_2.to_csv('../all_data/traj1-2.csv', index=False)
    
    traj_flow_3_5 = extract_traj('trajctory3-5.csv')
    traj_flow_3_5.dropna(subset = ['driver_id'], axis = 0, inplace = True)
    traj_flow_3_5.to_csv('../all_data/traj3-5.csv', index=False)