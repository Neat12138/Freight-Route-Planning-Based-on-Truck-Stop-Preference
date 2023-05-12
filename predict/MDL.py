from tqdm import tqdm
import objclass
import pandas as pd
import osmnx as ox
import numpy as np
import math
from collections import defaultdict
import copy
import pickle
import pygeohash as pgh

def get_lcs(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    result1=[]
    result2=[]
    dp_array = np.zeros((len_str1, len_str2),np.int8)
    # 初始化第一行和第一列
    value1 = str1[0].type
    value2 = str2[0].type
    for row in range(dp_array.shape[0]):
        if str1[row].type==value2:
            dp_array[row][0]=1

    for col in range(dp_array.shape[1]):
        if str2[col].type==value1:
            dp_array[0][col]=1
    max_length = 0
    for h in range(0, dp_array.shape[0]):
        for w in range(0, dp_array.shape[1]):
            if str1[h].type==str2[w].type:
                dp_array[h][w] = dp_array[h-1][w-1] + 1
                if max_length < dp_array[h][w]: 
                  if str1[h] not in result1:           
                    result1.append(str1[h])
                  if str2[w] not in result2:
                    result2.append(str2[w])
                max_length = max(max_length, dp_array[h][w])
            else:
                dp_array[h][w] = max(dp_array[h-1][w], dp_array[h][w-1])
    if len(result1) < len(result2):
      result2 = result2[:len(result1)]
    else:
      result1 = result1[:len(result2)]
    return result1, result2

#计算seq1和seq2的相似度
def simi(trip1, trip2):
    # 判断个数是否一致
    seq1, seq2 = get_lcs(trip1.sequence, trip2.sequence)
    if len(seq1) != len(seq2):
        return 0
    # 判断类别是否一致
    for sp1, sp2 in zip(seq1, seq2):
        if sp1.type != sp2.type:
            return 0
    if len(seq1) == len(seq2) == 0:
      return 0
    # 获取行驶时长
    dri_t1, dri_t2 = [], []
    t1_first = seq1[0].start_staytime - trip1.origin.departure_time
    t2_first = seq2[0].start_staytime - trip2.origin.departure_time
    dri_t1.append(t1_first.seconds)
    dri_t2.append(t2_first.seconds)
    for sp1_front, sp1_curr in zip(seq1[:-1], seq1[1:]):
        dri_t1.append((sp1_curr.start_staytime-sp1_front.start_staytime-pd.Timedelta(sp1_front.duration)).seconds)
    for sp2_front, sp2_curr in zip(seq2[:-1], seq2[1:]):
        dri_t2.append((sp2_curr.start_staytime-sp2_front.start_staytime-pd.Timedelta(sp2_front.duration)).seconds)
    t1_end = trip1.destination.arrive_time-seq1[-1].start_staytime-pd.Timedelta(seq1[-1].duration)
    t2_end = trip2.destination.arrive_time-seq2[-1].start_staytime-pd.Timedelta(seq2[-1].duration)
    dri_t1.append(t1_end.seconds)
    dri_t2.append(t2_end.seconds)
    # 获取停留时长
    stay1, stay2 = [], []
    for sp1, sp2 in zip(seq1, seq2):
        stay1.append(pd.Timedelta(sp1.duration).seconds)
        stay2.append(pd.Timedelta(sp2.duration).seconds)
    # 计算行驶时长相似性
    simi_dri = 0
    for t1, t2 in zip(dri_t1, dri_t2):
        simi_dri += (1-abs(t1-t2)/(t1+t2+1e-8))
    simi_dri = 2 * simi_dri/(len(trip1.sequence) + len(trip2.sequence) + 2)
    # 计算停留时长相似性
    simi_stay = 0
    for t1, t2 in zip(stay1, stay2):
        simi_stay += (1-abs(t1-t2)/(t1+t2+1e-8))
    simi_stay = 2 * simi_stay/(len(trip1.sequence) + len(trip2.sequence))
    # 总的相似度
    w_dri = (sum(dri_t1)+sum(dri_t2))/(sum(dri_t1)+sum(dri_t2)+sum(stay1)+sum(stay2))
    w_stay = (sum(stay1)+sum(stay2))/(sum(dri_t1)+sum(dri_t2)+sum(stay1)+sum(stay2))
    simi_all = w_dri*simi_dri + w_stay*simi_stay
    return simi_all

#找出历史数据中与当前数据出发时段与起终点在相同网格内的轨迹 
def heuristic_split_space_time(his_data, curr_data):  # , dist=10000
    # 按照出发时段过滤
    hour_split = defaultdict(list)
    for trip in his_data:
        h = trip.origin.departure_time.hour
        hour_split[h].append(trip)
    se_hour_split = hour_split[curr_data.origin.departure_time.hour]
    # 按地理空间距离过滤
    dist_split = []
    curr_ori_loc = (curr_data.origin.longitude, curr_data.origin.latitude)
    curr_dest_loc = (curr_data.destination.longitude, curr_data.destination.latitude)
    for trip in se_hour_split:
        trip_ori_loc = (trip.origin.longitude, trip.origin.latitude)
        trip_dest_loc = (trip.destination.longitude, trip.destination.latitude)
        # 判断起终点是否落在相同的网格内（由于目前都是从日照钢厂出发，暂时先不考虑起点差异性）
        tag_same_origin = False
        tag_same_destination = False
        if pgh.encode(curr_ori_loc[1], curr_ori_loc[0], precision=5) == \
                pgh.encode(trip_ori_loc[1], trip_ori_loc[0], precision=5):
            tag_same_origin = True
        if pgh.encode(curr_dest_loc[1], curr_dest_loc[0], precision=5) == \
                pgh.encode(trip_dest_loc[1], trip_dest_loc[0], precision=5):
            tag_same_destination = True
        if tag_same_destination:  # tag_same_origin and
            dist_split.append(trip)
    time_dist_split = dist_split
    return time_dist_split

def mdl_split_space_time(S, curr_data):
    hour = curr_data.origin.departure_time.hour
    se_hour_split = None
    for slot in S:
        if slot.left <= hour < slot.right:
            se_hour_split = list(slot.trip_set)
    dist_split = []
    curr_ori_loc = (curr_data.origin.longitude, curr_data.origin.latitude)
    curr_dest_loc = (curr_data.destination.longitude, curr_data.destination.latitude)
    for trip in se_hour_split:
        trip_ori_loc = (trip.origin.longitude, trip.origin.latitude)
        trip_dest_loc = (trip.destination.longitude, trip.destination.latitude)
        # 判断起终点是否落在相同的网格内（由于目前都是从日照钢厂出发，暂时先不考虑起点差异性）
        tag_same_origin = False
        tag_same_destination = False
        if pgh.encode(curr_ori_loc[1], curr_ori_loc[0], precision=5) == \
                pgh.encode(trip_ori_loc[1], trip_ori_loc[0], precision=5):
            tag_same_origin = True
        if pgh.encode(curr_dest_loc[1], curr_dest_loc[0], precision=5) == \
                pgh.encode(trip_dest_loc[1], trip_dest_loc[0], precision=5):
            tag_same_destination = True
        if tag_same_destination:  # tag_same_origin and
            dist_split.append(trip)    
    time_dist_split = dist_split
    return time_dist_split

# 获取相似度最高的候选位置
def find_simi_best(his_data):
    simi_candi = {}
    simi_bound = float('-inf')
    se_plan_no = None
    # 调度单号到行程的字典
    plan_no_trip_dict = {}
    for trip in his_data:
        plan_no_trip_dict[trip.plan_no] = trip
    for trip_i in his_data:
        for trip_j in his_data:
            # key1 = trip_i.plan_no + trip_j.plan_no
            # key2 = trip_j.plan_no + trip_i.plan_no
            # try:
            #   simi_value = trip_simi_dict[key1]
            # except Exception:
            #   if simi_value == 0:
            #     try:
            #       simi_value = trip_simi_dict[key2]
            #     except Exception:
            #       pass
            simi_value = simi(trip_i, trip_j)
            simi_candi.setdefault(trip_i.plan_no, []).append(simi_value)
            simi_candi.setdefault(trip_j.plan_no, []).append(simi_value)
    for plan_no, value in simi_candi.items():
        avg_simi = sum(value)/len(value)
        if avg_simi > simi_bound:
            simi_bound = avg_simi
            se_plan_no = plan_no
    return plan_no_trip_dict[se_plan_no], simi_bound

class Slot:
    def __init__(self, left, right, trip_set):
        self.left = left
        self.right = right
        self.trip_set = trip_set
        self.ent = self.get_ent()

    def get_ent(self):
        if len(self.trip_set) < 1:
            return float('inf')
        # elif len(self.trip_set) == 1:
        #     return {list(self.trip_set)[0].plan_no: 1}
        trip_list = list(self.trip_set)
        simi_ent = 0
        for trip_i in trip_list:
            for trip_j in trip_list:
              # key1 = trip_i.plan_no + trip_j.plan_no
              # key2 = trip_j.plan_no + trip_i.plan_no
              # try:
              #   simi_value = trip_simi_dict[key1]
              # except Exception:
              #   if simi_value == 0:
              #     try:
              #       simi_value = trip_simi_dict[key2]
              #     except Exception:
              #       pass
                simi_value = simi(trip_i, trip_j)
                simi_ent -= simi_value * math.log(simi_value + 1e-8)
        simi_ent /= math.pow(len(trip_list), 2)
        return simi_ent

# 计算MDL
class MDL:
    def __init__(self, nums=0, slots=None):
        self.nums = nums
        self.slots = slots

    def calcu(self):
        # MDL = L(H) + L(D|H)
        l_h = math.log(self.nums) + sum([math.log(slot.right-slot.left) for slot in self.slots])
        l_d_h = sum([int(math.log(len(slot.trip_set))) + slot.ent for slot in self.slots])
        return l_h + l_d_h

def get_geohash_trip(trip_set):
    geohash_dict = {}
    for trip in tqdm(trip_set):
        all_encode = get_encode(trip)
        geohash_dict.setdefault(all_encode, []).append(trip)
    with open('geohash_dict.pickle', 'wb') as f:
        pickle.dump(geohash_dict, f)

def get_encode(trip):
    trip_ori_loc = (trip.origin.longitude, trip.origin.latitude)
    trip_dest_loc = (trip.destination.longitude, trip.destination.latitude)
    ori_encode = pgh.encode(trip_ori_loc[1], trip_ori_loc[0], precision=5)
    dest_encode = pgh.encode(trip_dest_loc[1], trip_dest_loc[0], precision=5)
    all_encode = ori_encode + '_' + dest_encode
    return all_encode

def appro_partition(geohash_dict, q_trip):
    all_encode = get_encode(q_trip)
    if all_encode not in geohash_dict.keys():
        print(all_encode)
        return None
    trip_set = set(geohash_dict[all_encode])
    slot = Slot(left=0, right=24, trip_set=trip_set)
    S = [slot]
    mdl = MDL(nums=1, slots=S)
    minCost = mdl.calcu()
    while True:
        se_slot = None
        max_ent = float('-inf')
        for index, slot in enumerate(S):
            if slot.right - slot.left == 1:
                continue
            ent_value = slot.ent
            if ent_value > max_ent:
                max_ent = ent_value
                se_slot = slot
        split_index = None
        min_ent = float('inf')
        for i in range(se_slot.left+1, se_slot.right):
            sub_slot1_trip_set = slot_get_trip(se_slot.left, i, trip_set)
            sub_slot2_trip_set = slot_get_trip(i, se_slot.right, trip_set)
            sub_slot1 = Slot(se_slot.left, i, sub_slot1_trip_set)
            sub_slot2 = Slot(i, se_slot.right, sub_slot2_trip_set)
            new_ent = sub_slot1.ent + sub_slot2.ent
            if new_ent < min_ent:
                min_ent = new_ent
                split_index = i
        # 无法细化
        if min_ent == float('inf'):
            break
        sub_slot1_trip_set = slot_get_trip(se_slot.left, split_index, trip_set)
        sub_slot2_trip_set = slot_get_trip(split_index, se_slot.right, trip_set)
        sub_slot1 = Slot(se_slot.left, split_index, sub_slot1_trip_set)
        sub_slot2 = Slot(split_index, se_slot.right, sub_slot2_trip_set)
        new_S = copy.deepcopy(S)
        remove_index = S.index(se_slot)
        new_S.pop(remove_index)
        new_S.append(sub_slot1)
        new_S.append(sub_slot2)
        mdl = MDL(len(new_S), new_S)
        newCost = mdl.calcu()
        if newCost < minCost:
            minCost = newCost
            S = new_S
        else:
            break
    return S

def slot_get_trip(span_left, span_right, trip_set):
    new_trip_set = set()
    for trip in trip_set:
        hour = trip.origin.departure_time.hour
        if span_left <= hour < span_right:
            trip_set.add(trip)
    return new_trip_set

def calcu_test_simi(data_map_test, fixed_time=False):
    ans = []
    for index, q_trip in tqdm(enumerate(data_map_test)):
      if fixed_time:
          time_dist_split_trip = heuristic_split_space_time(data_map_train, q_trip)
      else:
          S = appro_partition(geohash_dict, q_trip)
          if S is None:
              continue
          time_dist_split_trip = mdl_split_space_time(S, q_trip)
      if len(time_dist_split_trip) < 1:
          ans.append((None, None, 0))
          continue
      if len(time_dist_split_trip) == 1:
          ans.append((time_dist_split_trip[0], q_trip, simi(time_dist_split_trip[0], q_trip)))
          continue
      match_trip, simi_bound = find_simi_best(time_dist_split_trip)
      ans.append((match_trip, q_trip, simi(match_trip, q_trip)))
    return ans

def find_matched_best_trip(query):
  S = appro_partition(geohash_dict, query)
  if S is None:
      return None
  time_dist_split_trip = mdl_split_space_time(S, query)
  if len(time_dist_split_trip) < 1:
    result = None
  if len(time_dist_split_trip) == 1:
    result = time_dist_split_trip[0]
  match_trip, simi_bound = find_simi_best(time_dist_split_trip)
  result = match_trip
  return result

def get_trip_simi(data_map_train):
    trip_simi_dict = {}
    index = 1
    for trip1 in tqdm(data_map_train[:-1]):
      for trip2 in data_map_train[index:]:
        key = trip1.plan_no + trip2.plan_no
        trip_simi_dict[key] = simi(trip1, trip2)
      index += 1
    with open('trip_simi_dict.pickle', 'wb') as f:
        pickle.dump(trip_simi_dict, f)
        
if __name__ == "__main__":
    with open('spseq_list.pickle', 'rb') as f:
        data_map = pickle.load(f)
    # data_map_train = data_map[:int(len(data_map)*0.99)]
    # data_map_test = data_map[int(len(data_map)*0.99):]
    # # q_train, seq_train = [], []
    # # q_test, seq_test = [], []
    # # print(len(data_map))
    # print(len(data_map_test))
    # # 保存起终点在相同网格内的数据
    # get_geohash_trip(data_map_train)
    with open('geohash_dict.pickle', 'rb') as f:
        geohash_dict = pickle.load(f)
    # # 保存两条轨迹之间的相似度
    get_trip_simi(data_map)
    # # with open('trip_simi_dict.pickle', 'rb') as f:
    # #     trip_simi_dict = pickle.load(f)
    ans = calcu_test_simi(data_map_test)