import pandas as pd
import numpy as np
from collections import defaultdict
import os
from objclass import SPSeq, StayPoint, Origin, Destination
from datetime import datetime
from tqdm import tqdm
import pickle
import scipy.cluster.hierarchy as sch
from utils import haversine_distance_2
import time


def get_spseq_list():
    # 每个停留点的数据
    sp_poi = pd.read_csv('../all_data/without_merge_sp_poi-11-12-2.csv')
    # 获取行程开始时间和结束时间
    path = '../all_data/'
    filename = 'traj11-12-2_preprocess.csv'
    traj_data = pd.read_csv(os.path.join(path, filename))
    traj_start = traj_data.drop_duplicates(subset='plan_no', keep='first')
    traj_end = traj_data.drop_duplicates(subset='plan_no', keep='last')
    traj_start_dict = {}
    for plan_no, value in traj_start.groupby('plan_no'):
        longitude = value['longitude'].values[0]
        latitude = value['latitude'].values[0]
        traj_start_dict[plan_no] = (longitude, latitude)
    traj_end_dict = {}
    for plan_no, value in traj_end.groupby('plan_no'):
        longitude = value['longitude'].values[0]
        latitude = value['latitude'].values[0]
        traj_end_dict[plan_no] = (longitude, latitude)


    spseq_list = []
    for plan_no, seq in tqdm(sp_poi.groupby('plan_no')):
        # if plan_no not in set(traj_start['plan_no']):
        #     continue

        dri_id = seq['dri_id'].values[0]
        # 获取起点位置、出发时间
        start_time = traj_start[traj_start['plan_no'] == plan_no]['time'].values[0]
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        start_lng, start_lat = traj_start_dict[plan_no]
        origin = Origin(start_lng, start_lat, start_time)
        # 获取终点位置、到达时间
        end_time = traj_end[traj_end['plan_no'] == plan_no]['time'].values[0]
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        end_lng, end_lat = traj_end_dict[plan_no]
        destination = Destination(end_lng, end_lat, end_time)

        spseqobj = SPSeq(plan_no, dri_id, origin, destination)
        for index, sp in seq.iterrows():
            cen_lng = sp['longitude']
            cen_lat = sp['latitude']
            start_staytime = sp['start_stay']
            start_staytime = datetime.strptime(start_staytime, "%Y-%m-%d %H:%M:%S")
            duration = sp['duration']
            dist = sp['dist']
            spobj = StayPoint(cen_lng, cen_lat, start_staytime, duration, plan_no, dri_id, dist=dist)
            spobj.type = sp['poi_type']
            spseqobj.sequence.append(spobj)
        spseq_list.append(spseqobj)

    with open('spseq_raw_list11-12-2.pickle', 'wb') as f:
        pickle.dump(spseq_list, f)
    with open('spseq_raw_list11-12-2.pickle', 'rb') as f:
        spseq_list = pickle.load(f)
    return spseq_list


def get_dest_id(spseq_list):
    # 需要根据上下文来找规律，即起点、终点、出发时段
    # 停留偏好能够预测的依据：在一个严格的上下文下，停留偏好序列近似
    # 对所有行程终点进行聚类，层次聚类
    print(len(spseq_list))
    dest_list = []
    for index, seqobj in tqdm(enumerate(spseq_list)):
        lon, lat = seqobj.destination.longitude, seqobj.destination.latitude
        dest_list.append([lon, lat])
    t0 = time.time()
    # 计算距离矩阵
    dis = sch.distance.pdist(dest_list, haversine_distance_2)  # 计算点与点距离
    z = sch.linkage(dis, method='average')  # 得到矩阵z
    dest_cluster_list = sch.fcluster(z, t=2000, criterion='distance')  # 层次聚类
    with open('dest_cluster_list11-12-2.pickle', 'wb') as f:
        pickle.dump(dest_cluster_list, f)
    with open('dest_cluster_list11-12-2.pickle', 'rb') as f:
        dest_cluster_list = pickle.load(f)
    t1 = time.time()
    print(t1-t0, 's')
    print(dest_cluster_list)
    for seqobj, dest_cluster_id in tqdm(zip(spseq_list, dest_cluster_list)):
        seqobj.dest_cluster_id = dest_cluster_id
    return spseq_list


def remove_short_time_sp(spseq_list):
    # 删除停留时长小于20分钟情况
    for seqobj in tqdm(spseq_list):
        new_sequence = []
        for index, sp in enumerate(seqobj.sequence):
            if pd.Timedelta(sp.duration).seconds/60 >= 20:
                new_sequence.append(sp)
        seqobj.sequence = new_sequence
    # 删除停留点间隔大于4小时情况
    new_spseq_list = []

    num_null = 0
    for seqobj in tqdm(spseq_list):
        if len(seqobj.sequence) == 0:
            num_null += 1
            continue

        origin = seqobj.origin
        dest = seqobj.destination
        flag = True
        for index, sp in enumerate(seqobj.sequence):
            if index == 0:
                if (sp.start_staytime - origin.departure_time).seconds > 4*60*60:
                    flag = False
            else:
                if (sp.start_staytime - last_sp.start_staytime).seconds > 4*60*60:
                    flag = False
            last_sp = sp
        if (dest.arrive_time - seqobj.sequence[-1].start_staytime).seconds > 4*60*60:
            flag = False
        if flag is True:
            new_spseq_list.append(seqobj)
    spseq_list = new_spseq_list
    print(num_null)
    return spseq_list


def static_spseq(spseq_list):
    # 划分时段、司机编号、地址编码进行统计
    split_dest_time = dict()
    split_dest_dri = dict()
    split_dest_dri_time = dict()
    split_dest_time_dri = dict()
    for seqobj in spseq_list:
        start_hour = seqobj.origin.departure_time.hour
        dri_id = seqobj.dri_id
        dest_id = seqobj.dest_cluster_id
        split_dest_time.setdefault(dest_id, {})
        split_dest_time[dest_id].setdefault(start_hour, []).append(seqobj)
        split_dest_dri.setdefault(dest_id, {})
        split_dest_dri[dest_id].setdefault(dri_id, []).append(seqobj)

        split_dest_dri_time.setdefault(dest_id, {})
        split_dest_dri_time[dest_id].setdefault(dri_id, {})
        split_dest_dri_time[dest_id][dri_id].setdefault(start_hour, []).append(seqobj)

        split_dest_time_dri.setdefault(dest_id, {})
        split_dest_time_dri[dest_id].setdefault(start_hour, {})
        split_dest_time_dri[dest_id][start_hour].setdefault(dri_id, []).append(seqobj)

    return split_dest_time, split_dest_dri, split_dest_dri_time, split_dest_time_dri


def show_split_dest_time(split_dest_time):
    for dest_id, split_time in split_dest_time.items():
        print("终点：", dest_id)
        for start_time, seqobjlist in split_time.items():
            print("开始时间：", start_time, "停留序列个数：", len(seqobjlist))
            if len(seqobjlist) < 100:
                continue
            poi_dict = defaultdict(int)
            for seqobj in seqobjlist:
                poi_seq = ' & '.join(seqobj.get_seq_class())
                poi_stay = ' & '.join(seqobj.get_seq_duration())
                poi_dict[poi_seq] += 1
                print(seqobj.plan_no, seqobj.origin.departure_time, seqobj.destination.arrive_time)
                print(poi_seq, '----', poi_stay)
            poi_dict = sorted(poi_dict.items(), key=lambda x: x[1], reverse=True)
            print(poi_dict)
            print()


def show_split_dest_time_dri(split_dest_time_dri):
    for dest_id, split_time_dri in split_dest_time_dri.items():
        print()
        print("终点：", dest_id)
        for start_time, split_dri in split_time_dri.items():
            print("开始时间：", start_time)
            for dri_id, seqobjlist in split_dri.items():
                print("司机ID：", dri_id, "停留序列个数：", len(seqobjlist))
                if len(seqobjlist) < 10:
                    continue
                poi_dict = defaultdict(int)
                for seqobj in seqobjlist:
                    poi_seq = ' & '.join(seqobj.get_seq_class())
                    poi_stay = ' & '.join(seqobj.get_seq_duration())
                    poi_dict[poi_seq] += 1
                    print(seqobj.plan_no, seqobj.origin.departure_time, seqobj.destination.arrive_time)
                    print(poi_seq, '----', poi_stay)
                poi_dict = sorted(poi_dict.items(), key=lambda x: x[1], reverse=True)
                print(poi_dict)
                print()
                
if __name__ == '__main__':
    spseq_list = get_spseq_list()
    spseq_list = get_dest_id(spseq_list)
    spseq_list = remove_short_time_sp(spseq_list)
    with open('spseq_list11-12-2.pickle', 'wb') as f:
        pickle.dump(spseq_list, f)
