import numpy as np
import pandas as pd
import datetime
import utils


class TrajPoint:
    def __init__(self, plan_no, longitude, latitude, time, dist=None):
        self.plan_no = plan_no
        self.longitude = longitude
        self.latitude = latitude
        self.time = time
        self.dist = dist  # 离起点距离

class Traj:
    def __init__(self, plan_no):
        self.plan_no = plan_no
        # self.waybill_no = waybill_no
        self.dri_id = None
        self.return_loc = None
        self.create_time = None
        self.load_time = None
        self.return_time = None
        self.traj_point_list = []

# 存放所有轨迹
class TrajAll:
    def __init__(self):
        self.traj_dict = {}

# 停留点类
class StayPoint:
    def __init__(self, cen_lng, cen_lat, start_staytime, duration, plan_no, dri_id, popularity, end_point=None, position=None,
                 staypoilist=None, cluster_id=None, dist=None, matched_road_id=None):
        self.sp_id = None
        self.cen_lng = cen_lng  # 中心点经度 float
        self.cen_lat = cen_lat  # 中心点纬度 float
        self.start_staytime = start_staytime  # 停留发生时间 datetime
        self.duration = duration  # 停留时间 float
        self.dist = dist  # 距起点距离
        # self.waybill_no = waybill_no  # 运单号
        self.plan_no = plan_no  # 调度单号
        self.dri_id = dri_id  # 司机ID
        self.end_point = end_point  # 对应运输终点编号
        self.staypoilist = staypoilist  # 低速轨迹点列表 list
        self.cluster_id = cluster_id  # 聚类编号
        self.matched_road_id = matched_road_id
        self.type = None #停留点类型
        self.position = position  # 在轨迹中的位次
        self.sub_traj_point_list = None  # 存放子轨迹段
        self.popularity = popularity


# 停留区域
class StayRegion:
    def __init__(self, sr_id, region_type, num_staypoint, popularity, region_node, cen_lng=None, cen_lat=None, staypoiobj_list = None, planno_set = None):
        self.sr_id = sr_id  # 地点ID
        self.cen_lng = cen_lng  # 区域的中心点经度
        self.cen_lat = cen_lat  # 区域的中心点纬度
        self.staypoiobj_list = staypoiobj_list  # 该区域中停留点实体(staypoi类的实体)的列表
        self.trip_set = set()
        self.vertex_list = []  # 多边形顶点的列表 存放Point
        self.region_type = region_type  # 停留区域类型
        self.convex_hull = None  # convex_hull
        self.polygon = None
        self.feature = []  # 特征向量
        self.num_staypoint = num_staypoint  # 停留点个数
        self.nearRoad_type = None  # string
        self.nearPoiType_distribute = None  # list
        self.area = None  # 区域面积 float
        self.turn_position = None  # 转向位置
        self.pid = {}  # 统计pid出现过的次数
        self.planno_set = planno_set  # 存放调度单的集合，防止被重复传入
        self.score = 0  # 计算候选位置得分
        self.cargo_type_set = set()  # 获取货物类型
        self.time_distribution = None  # 访问时段分布
        self.avg_duration = None  # 访问时长分布
        self.stop_fre = None  # 停留频次分布
        self.dist_end = {}  # 离实际终点距离, key: 终点ID, value: 离终点距离
        self.trip_converage = {}  # 行程覆盖率
        self.popularity = popularity
        self.region_node = region_node

    def get_center_loc(self):  # 得到停留区域中心坐标
        points = []
        for sp in self.staypoiobj_list:
            points.append((sp.cen_lng, sp.cen_lat))
        points = np.array(points)
        self.cen_lng, self.cen_lat = np.mean(points, axis=0)


class Point:
    def __init__(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude


class Origin(Point):
    def __init__(self, longitude, latitude, departure_time):
        super().__init__(longitude, latitude)
        self.departure_time = departure_time


class Destination(Point):
    def __init__(self, longitude, latitude, arrive_time):
        super().__init__(longitude, latitude)
        self.arrive_time = arrive_time


# 停留点序列
class SPSeq:
    def __init__(self, plan_no, dri_id, origin, destination):
        self.plan_no = plan_no
        self.dri_id = dri_id
        self.origin = origin
        self.destination = destination
        self.sequence = []  # 存放停留点对象

    def get_stay_time(self):
        return len(self.sequence)

    def get_seq_class(self):
        ans = []
        for sp in self.sequence:
            ans.append(sp.type)
        return ans

    def get_seq_duration(self):
        ans = []
        for sp in self.sequence:
            ans.append(sp.duration)
        return ans

    def get_seq_dist(self):
        ans = []
        for sp in self.sequence:
            ans.append(sp.dist)
        return ans

    def get_seq_continue_time(self):
        ans = []
        for index, sp in enumerate(self.sequence):
            if index == 0:
                continue_time = sp.duration
            else:
                continue_time = sp.start_time - last_sp.time - last_sp.duration
            last_sp = sp
            ans.append(continue_time)
        return ans

    def get_seq_all(self):
        ans = []
        for index, sp in enumerate(self.sequence):
            if index == 0:
                continue_time = sp.duration
            else:
                continue_time = sp.start_time - last_sp.time - last_sp.duration
            last_sp = sp
            ans.append([sp.type, sp.start_time, continue_time, sp.duration, sp.dist])
        return ans

# 记录地址类
class Address:
    def __init__(self, end_point, lab_lng, lab_lat, tag_lng, tag_lat):
        self.end_point = end_point
        self.lab_lng = lab_lng  # 地址库记录位置
        self.lab_lat = lab_lat
        self.tag_lng = tag_lng  # 标注位置
        self.tag_lat = tag_lat
        self.candiates_list = []  # 记录候选位置
        self.candiates_label_index = None  # 记录候选位置标签
        self.trip_set = set()
        self.trip_list = []  # 存放每个轨迹对象
        self.trip_converage = None
        self.distance = None
        self.avg_duration = None
        self.num_dri = None
        self.time_distribution = None
        self.num_deliveries = None
        self.nearPoiType_distribute = None
        self.address_text = None

    def get_trip_converage(self):
        trip_coverage = []
        for candiobj in self.candiates_list:
            coverage = len(self.trip_set.intersection(candiobj.trip_set)) / len(self.trip_set)
            trip_coverage.append(coverage)
        self.trip_converage = trip_coverage

    def get_distance(self):
        distance = []
        for candiobj in self.candiates_list:
            p1_lng, p1_lat = candiobj.cen_lng, candiobj.cen_lat
            p2_lng, p2_lat = self.lab_lng, self.lab_lat
            dist = utils.haversine_distance_loc_points(p1_lat, p1_lng, p2_lat, p2_lng)
            distance.append(dist)
        self.distance = distance

    def get_avg_duration(self):
        avg_duration = []
        for candiobj in self.candiates_list:
            tmp_dur = 0
            for spobj in candiobj.staypoiobj_list:
                duration = pd.to_timedelta(spobj.duration).total_seconds()//60
                tmp_dur += duration
            tmp_dur /= len(candiobj.staypoiobj_list)
            avg_duration.append(tmp_dur)
        self.avg_duration = avg_duration

    def get_num_dri(self):
        num_dri = []
        for candiobj in self.candiates_list:
            dri_set = set()
            for spobj in candiobj.staypoiobj_list:
                dri_set.add(spobj.dri_id)
            num_dri.append(len(dri_set))
        self.num_dri = num_dri

    def get_time_distribution(self):
        time_distribution = []
        for candiobj in self.candiates_list:
            tmp_time_distribution = [0]*24
            for spobj in candiobj.staypoiobj_list:
                timestamp = datetime.datetime.strptime(spobj.start_staytime, '%Y-%m-%d %H:%M:%S').hour # 获得小时
                tmp_time_distribution[timestamp] += 1
            time_distribution.append(np.array(tmp_time_distribution)/sum(tmp_time_distribution))
        self.time_distribution = time_distribution

    def get_num_deliveries(self):
        deliveries_set = set()
        for candiobj in self.candiates_list:
            for spobj in candiobj.staypoiobj_list:
                if spobj.end_point == self.end_point:
                    deliveries_set.add(spobj.plan_no)
        self.num_deliveries = len(deliveries_set)




class SubGrid:
    def __init__(self, max_lng, max_lat, min_lng, min_lat):
        self.max_lng = max_lng
        self.max_lat = max_lat
        self.min_lng = min_lng
        self.min_lat = min_lat
        self.ptcnt = 0

    def contains(self, lng, lat):
        if self.min_lat <= lat <= self.max_lat and self.min_lng <= lng <= self.max_lng:
            return True
        else:
            return False

    def get_center(self):
        return (self.max_lng + self.min_lng) / 2, (self.max_lat + self.min_lat) / 2

class Grid:
    def __init__(self, max_lng, max_lat, min_lng, min_lat, grid_num):
        self.max_lng = max_lng
        self.max_lat = max_lat
        self.min_lng = min_lng
        self.min_lat = min_lat
        self.subgrids = self.create_grid(self.max_lng, self.max_lat, self.min_lng, self.min_lat, grid_num)

    def create_grid(self, max_lng, max_lat, min_lng, min_lat, grid_num):
        lng_list = np.linspace(min_lng, max_lng, grid_num+1)
        lat_list = np.linspace(min_lat, max_lat, grid_num+1)
        lng_interval = (max_lng-min_lng) / grid_num
        lat_interval = (max_lat-min_lat) / grid_num
        subgridlist = []
        for lat in lat_list[:-1]:
            for lng in lng_list[:-1]:
                subgrid = SubGrid(lng+lng_interval, lat+lat_interval, lng, lat)
                subgridlist.append(subgrid)
        return subgridlist

    def insert_point(self, lng, lat):
        for subgrid in self.subgrids:
            if subgrid.contains(lng, lat):
                subgrid.ptcnt += 1
                break

    def find_max_grid(self):
        max_pt = float('-inf')
        max_density_subgrid = None
        for subgrid in self.subgrids:
            if max_pt < subgrid.ptcnt:
                max_pt = subgrid.ptcnt
                max_density_subgrid = subgrid
        return max_density_subgrid.get_center()

class ReturnCluster:
    def __init__(self, lng, lat, label_id, cluster_num):
        self.lng = lng
        self.lat = lat
        self.label_id = label_id
        self.cluster_num = cluster_num


class ReturnCandidate:
    def __init__(self, lng, lat):
        self.lng = lng
        self.lat = lat
        self.id = None  # 记录返单位置ID
        self.cluster_label_id = None
        self.is_best = False
        self.cluster_label_density = None  # 所在簇密度
        self.dist2max_density_cluster = None  # 离最大簇距离
        self.dist2tag_loc = None  # 距离标注位置距离
        self.knn_dist = None  # 离k个位置的平均距离
        self.dist2road = None  # 距离道路距离

class AddContext:
    def __init__(self):
        self.delivery_num = None  # 过去交付数量
        self.median_pair_dist = None
        self.P10_pair_dist = None
        self.density = None  # 该地址对应密度