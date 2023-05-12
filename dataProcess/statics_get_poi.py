import pandas as pd
import utils
from requests_html import HTMLSession
import json
from tqdm import tqdm


def add_label(sp_data):
    # 给每个停留点打上cluster_id (cluster_id: 新生成的唯一ID, label: 簇的ID)
    label2id = []
    num_negative = 0
    for index, value in sp_data.iterrows():
        if value['label'] == -1:
            label2id.append(str(num_negative)+'_-1')
            num_negative += 1
        else:
            label2id.append(str(value['label']))
    label2id = pd.DataFrame(data=label2id, columns=['cluster_id'])
    sp_data_add_id = pd.concat([sp_data, label2id], axis=1)
    return sp_data_add_id


def get_cluster_location(sp_data_add_id):
    label_negative = sp_data_add_id[sp_data_add_id['label'] == -1]
    label_non_negative = sp_data_add_id[sp_data_add_id['label'] != -1]
    label_non_negative = label_non_negative.drop_duplicates(subset='label', keep='first')
    label_all = pd.concat([label_negative, label_non_negative])
    label_all.reset_index(drop=True, inplace=True)
    # 查找每个停留点对应的POI
    label_all['location_gcj'] = label_all.apply(utils.wgs84togaode_arr, axis=1,
                                                args=('longitude', 'latitude'))
    return label_all


def get_poi(radius: int, data, keys: list):
    ans = {}
    radius_list = [radius]
    keys_index = 0
    for index, center_point in tqdm(data.iterrows()):
        if index <= 16660:
            continue
        for num in radius_list:
            PoiTypes = ['010100']  # '050000', '180300'
            key = keys[keys_index]
            for PoiType in PoiTypes:
                params = {
                    "key": key,
                    "location": [str(round(getattr(center_point, 'location_gcj')[0], 6))+','+\
                                 str(round(getattr(center_point, 'location_gcj')[1], 6))],
                    "types": PoiType,
                    "radius": num,
                    "output": "json",
                }
                url = 'https://restapi.amap.com/v3/place/around'
                session = HTMLSession()

                rq = session.get(url, params=params)
                result = json.loads(rq.html.html)
                status = result['status']

                if status == '0':
                    ans = pd.DataFrame(ans)
                    return ans

                total_page = result['count']
                ans.setdefault(str(PoiType) + '_' + str(num), []).append(total_page)

            cluster_id = str(getattr(center_point, 'cluster_id'))
            ans.setdefault('cluster_id', []).append(cluster_id)
    ans = pd.DataFrame(ans)
    return ans

if __name__ == '__main__':
    # 高德接口
    keys = ['your key']

    sp_data_file = '../all_data/cluster_sp_11-12-2.csv'
    sp_data = pd.read_csv(sp_data_file)
    # 查看一共有多少停留点
    print(len(sp_data))

    sp_data_add_id = add_label(sp_data)
    label_all = get_cluster_location(sp_data_add_id)
    # 查看一共有多少停留热点+未聚类成簇的停留点
    print(len(set(label_all['cluster_id'])))
    data = label_all[['location_gcj', 'cluster_id']]
    radius = 80
    ans = get_poi(radius, data, keys)

    PoiTypes = ['010100']  # '050000', '180300'
    names = ['汽车服务_加油站']  # '餐饮服务', '道路附属设施_服务区'
    radius_all = [80]
    columns = {}
    for r in radius_all:
        for name, PoiType in zip(names, PoiTypes):
            columns[PoiType+'_'+str(r)] = name+'_'+str(r)

    ans.rename(columns=columns, inplace=True)
    print(ans.head())
    ans.to_csv('../all_data/ans_poi-11-12-2.csv', index=False)
    sp_poi = pd.merge(label_all, ans, on='cluster_id', how='left')
    sp_poi.to_csv('../all_data/sp_poi-11-12-2.csv', na_rep='NULL', index=False)
    print(sp_poi.head())