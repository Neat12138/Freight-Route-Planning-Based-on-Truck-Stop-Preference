import pickle
import pandas as pd
import utils
from tqdm import tqdm
from objclass import StayPoint, StayRegion
import pandas as pd
from sklearn.cluster import DBSCAN

def do_DBSCAN_for_staypoints(spobj_list, eps, min_sample):
    cenpoi_list = [utils.wgs84_to_mercator(spobj.cen_lng, spobj.cen_lat) for spobj in spobj_list]
    labels = DBSCAN(eps=eps, min_samples=min_sample, metric='euclidean').fit(cenpoi_list).labels_
    for i in range(0, len(labels)):
        spobj_list[i].cluster_id = labels[i]
    return spobj_list, labels

def get_stay_region_with_popularity(miniclus, labels):
    label_set = set(labels)
    for sp, label in zip(miniclus, labels):
        sp.label = label
    
    stay_region = {}
    for label in label_set:
        if label != -1:
            spobj_list = []
            stay_region[label] = spobj_list

    stay_region_all = []
    for sp in miniclus:
        if sp.label != -1:
            stay_region[sp.label].append(sp)
        #为聚类成簇则单点为一个区域
        else:
            sr = StayRegion(sr_id = -1, cen_lng = sp.cen_lng, cen_lat = sp.cen_lat, region_type = sp.type, num_staypoint = 1, 
                         popularity = 1, staypoiobj_list = [sp], planno_set = set(sp.plan_no))
            stay_region_all.append(sr)
    #聚类成簇则进一步计算区域中心点和热门度
    for label in stay_region:
        cen_lng = 0
        cen_lat = 0
        plan_list = []
        for sp in stay_region[label]:
            region_type = sp.type
            cen_lng += sp.cen_lng
            cen_lat += sp.cen_lat
            plan_list.append(sp.plan_no)
        cen_lng /= len(stay_region[label])
        cen_lat /= len(stay_region[label])
        num_staypoint = len(stay_region[label])
        plan_set = set(plan_list)

        sr = StayRegion(sr_id = label, cen_lng = cen_lng, cen_lat = cen_lat, region_type = region_type, num_staypoint = num_staypoint, 
                         popularity = len(plan_set), staypoiobj_list = stay_region[label], planno_set = plan_set)
        stay_region_all.append(sr)
    return stay_region_all

if __name__ == '__main__':
    with open('spseq_list.pickle', 'rb') as f:
        spseq_list = pickle.load(f)
    rest_spobj_list = []
    gas_spobj_list = []
    restaurant_spobj_list = []
    for seq in spseq_list:
        for sp in seq.sequence:
            if sp.type == 'rest_area':
                rest_spobj_list.append(sp)
            elif sp.type == 'gas':
                gas_spobj_list.append(sp)
            elif sp.type == 'restaurant':
                restaurant_spobj_list.append(sp)
    gas_miniclus, gas_labels = do_DBSCAN_for_staypoints(gas_spobj_list, eps=200, min_sample=2)
    rest_miniclus, rest_labels = do_DBSCAN_for_staypoints(rest_spobj_list, eps=200, min_sample=3)
    restaurant_miniclus, restaurant_labels = do_DBSCAN_for_staypoints(restaurant_spobj_list, eps=200, min_sample=2)

    gas_stay_region = get_stay_region_with_popularity(gas_miniclus, gas_labels)
    rest_stay_region = get_stay_region_with_popularity(rest_miniclus, rest_labels)
    restaurant_stay_region = get_stay_region_with_popularity(restaurant_miniclus, restaurant_labels)

    all_stay_region = gas_stay_region + rest_stay_region + restaurant_stay_region
    with open('stay_region.pickle', 'wb') as f:
        pickle.dump(all_stay_region, f)

