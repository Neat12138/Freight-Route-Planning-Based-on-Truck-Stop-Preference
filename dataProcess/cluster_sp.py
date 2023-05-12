import pandas as pd
from sklearn.cluster import DBSCAN
from objclass import StayPoint
import utils
from objclass import TrajPoint
from time import time


def write2csv(ans, labels, save_path):
    sp_list = []
    for sp, label in zip(ans, labels):
        lng = sp.cen_lng
        lat = sp.cen_lat
        start = sp.start_staytime
        duration = sp.duration
        dist = sp.dist
        plan_no = sp.plan_no
        dri_id = sp.dri_id
        position = sp.position
        sp_list.append([plan_no, dri_id, lng, lat, start, duration, dist, position, label])
    sp_df = pd.DataFrame(data=sp_list,
                         columns=['plan_no', 'dri_id', 'longitude', 'latitude', 'start_stay',
                                  'duration', 'dist', 'position', 'label'])
    try:
        sp_df.to_csv(save_path, index=False)
    except Exception as e:
        print(e)


def do_DBSCAN_for_staypoints(spobj_list, eps=5, min_sample=5):
    cenpoi_list = [utils.wgs84_to_mercator(spobj.cen_lng, spobj.cen_lat) for spobj in spobj_list]
    labels = DBSCAN(eps=eps, min_samples=min_sample, metric='euclidean').fit(cenpoi_list).labels_
    for i in range(0, len(labels)):
        spobj_list[i].cluster_id = labels[i]
    return spobj_list, labels

if __name__ == '__main__':
    t0 = time()
    sp_data = pd.read_csv('../all_data/staypoint_11-12-2.csv')
    spobj_list = []
    for plan_no, sp_value in sp_data.groupby('plan_no'):
        sp_value.reset_index(drop=True, inplace=True)
        for index, sp_pre_plan in sp_value.iterrows():
            spobj_list.append(StayPoint(plan_no=sp_pre_plan['plan_no'], dri_id=sp_pre_plan['dri_id'],
                                        cen_lng=sp_pre_plan['longitude'], cen_lat=sp_pre_plan['latitude'],
                                        start_staytime=sp_pre_plan['start_stay'], duration=sp_pre_plan['duration'],
                                        dist=sp_pre_plan['dist'], end_point=None, position=sp_pre_plan['position']))
    # 对停留点进行聚类
    miniclus, labels = do_DBSCAN_for_staypoints(spobj_list, eps=50, min_sample=5)
    write2csv(miniclus, labels, '../all_data/cluster_sp_11-12-2.csv')
    t1 = time()
    print(t1-t0, 's')