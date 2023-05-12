#### 文件结构：

```
│  README.md
│
├─dataProcess
│      calculate_dist.py	# 计算与origin的距离
│      cluster_sp.py	# 停留点聚类
│      data_denosing.py	# 数据去噪
│      extract_stay_point.py	# 提取停留点
│      get_stay_region.py	# 提取停留区域
│      judge_poi_type.py	# 判断停留点类型
│      objclass.py	
│      read_data.py	# 读取实验数据
│      stastics_poi_seq.py	# 统计停留点序列
│      statics_get_poi.py	# 调用高德地图API
│      utils.py
│
├─predict
│      MDL.py	# 使用geohash和MDL准则划分相似轨迹数据
│      objclass.py
│      utils.py
│
└─routePlanning
        objclass.py
        ROSE.py	# 路径规划
        utils.py
```

