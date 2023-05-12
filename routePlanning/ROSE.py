import utils
import pickle
import osmnx as ox
from queue import Queue
import networkx as nx
import time

class Query:
  def __init__(self, popularity_threshold, origin, destination, origin_node, destination_node, poi_seq, poi_time_interval = None):
    self.popularity_threshold = popularity_threshold
    self.origin = origin
    self.destination = destination
    self.origin_node = origin_node
    self.destination_node = destination_node
    self.poi_seq = poi_seq
    self.poi_time_interval = poi_time_interval

class Path:
  def __init__(self, start, end, start_node, end_node, weight):
    self.start = start
    self.end = end
    self.start_node = start_node
    self.end_node = end_node
    self.weight = weight

class Candidate:
  def __init__(self, sr_id, paths):
    self.sr_id = sr_id
    self.paths = paths

  def dist_sum(self):
    self.dist = 0
    for path in self.paths:
      self.dist += path.weight
    return self.dist

def get_poi_embedding(popularity_threshold, all_region):
  stay_region = []
  for sr in all_region:
    if sr.popularity > popularity_threshold:
      stay_region.append(sr)
  for sr in stay_region:
    sr_node = ox.distance.nearest_nodes(G, sr.cen_lng, sr.cen_lat, return_dist = False)
    sr.region_node = sr_node
  embedding = []    
  rest_embedding = []
  gas_embedding = []
  restaurant_embedding = [] 
  for sr in stay_region:
    for next_sr in stay_region:
      if next_sr.sr_id != sr.sr_id:
        dist = utils.haversine_distance_loc_points(sr.cen_lng, sr.cen_lat,
                              next_sr.cen_lng, next_sr.cen_lat)
        if next_sr.sr_id[0] == '0':
          rest_embedding.append(Path(start = sr, end = next_sr, start_node = sr.region_node, end_node = next_sr.region_node, weight = dist))
        elif next_sr.sr_id[0] == '1':
          restaurant_embedding.append(Path(start = sr, end = next_sr, start_node = sr.region_node, end_node = next_sr.region_node, weight = dist))
        else:
          gas_embedding.append(Path(start = sr, end = next_sr, start_node = sr.region_node, end_node = next_sr.region_node, weight = dist))

  embedding.append(rest_embedding)
  embedding.append(restaurant_embedding)
  embedding.append(gas_embedding)
  return embedding, stay_region

def get_first_level(query, embedding):
  first_level = []
  visited_node = []
  first_category = query.poi_seq[0]
  first_time_interval = query.poi_time_interval[0]

  for path in embedding[first_category]:
    curr_path = []
    if 116.8 < path.start.cen_lng < 119.5:
      if 34.9 < path.start.cen_lat < 36.3:
        if path.start.sr_id not in visited_node:
          try:
            dist = nx.shortest_path_length(simplified_G, query.origin_node, path.start_node, weight = 'length')
          except Exception:
            dist = 99999999
          if 99999999 > dist > first_time_interval * 5:
            curr_path.append(Path(start = query.origin, end = path.start, start_node = query.origin_node,
                                  end_node = path.start_node, weight = dist))
            first_level.append(Candidate(path.start.sr_id, curr_path))
            visited_node.append(path.start.sr_id)
  return first_level

def get_final_level(query, embedding):
  final_level = []
  visited_node = []
  final_category = query.poi_seq[-1]
  for path in embedding[final_category]:
    if path.end.sr_id not in visited_node:
      if 116.8 < path.end.cen_lng < 119.5:
        if 34.9 < path.end.cen_lat < 36.3:
          try:
            dist = nx.shortest_path_length(simplified_G, path.end_node, query.destination_node, weight = 'length')
          except Exception:
            dist = 99999999
          if dist < 99999999:
            final_level.append(Path(start = path.end, end = query.destination, start_node = path.end_node,
                                  end_node = query.destination_node, weight = dist))
          visited_node.append(path.end.sr_id)
  return final_level

def get_guide_path(query, embedding, first_level, final_level, cost_tresh):

  all_candidates = []
  guide_path = []
  all_candidates.append(first_level)
  if len(query.poi_seq) > 1:
    for i in range(1, len(query.poi_seq)):
      curr_level = []
      curr_visited = []
      category = query.poi_seq[i]  
      time_interval = query.poi_time_interval[i-1]    
      #对于上一层访问过的节点
      for curr_candidate in all_candidates[-1]:
        for sr_embedding in embedding[category]:
          #查找当前点可选择的下一个停留点
             if sr_embedding.start.sr_id == curr_candidate.sr_id:
                if sr_embedding.weight > time_interval * 5:
                #若下一个停留点不存在于curr_level
                  if sr_embedding.end.sr_id not in curr_visited:
                      curr_visited.append(sr_embedding.end.sr_id)
                      curr_path = curr_candidate.paths.copy()
                      curr_path.append(sr_embedding)
                      curr_level.append(Candidate(sr_embedding.end.sr_id, curr_path))
                  else:
                    for next_candidate in curr_level:
                      if next_candidate.sr_id == sr_embedding.end.sr_id:
                          if next_candidate.dist_sum() > curr_candidate.dist_sum() + sr_embedding.weight:
                            new_path = curr_candidate.paths.copy()
                            new_path.append(sr_embedding)
                            next_candidate.paths = new_path
      all_candidates.append(curr_level)

  
  curr_level = []
  for candidate in all_candidates[-1]:
    for end_path in final_level:
      if candidate.sr_id == end_path.start.sr_id:
        final_dist = end_path.weight
        new_path = candidate.paths.copy()
        new_path.append(end_path)
        curr_level.append(Candidate(-1, new_path))
  all_candidates.append(curr_level)

  for candidate in all_candidates[-1]:
    if candidate.dist_sum() <= cost_tresh:
      guide_path = candidate.paths
      cost_tresh = candidate.dist_sum()

  return guide_path, cost_tresh

def update_path(G, embedding, query, first_level, final_level):
  exact_path_length = 9999999
  guide_path_length = 0
  while(exact_path_length != guide_path_length):
    print("guide path length", guide_path_length)
    print("exact path length", exact_path_length)
    guide_path, cost_tresh = get_guide_path(query, embedding, first_level, final_level, exact_path_length)
    guide_path_length = cost_tresh
    # if cost_tresh == exact_path_length:
    #   break
    nodes = []
    for path in guide_path:
      nodes.append(path.start_node)
    nodes.append(query.destination_node)

    index = 1
    exact_path_length = 0
    for path in guide_path[1:-1]:
      exact_path_length = guide_path[0].weight + guide_path[-1].weight
      try:
        path.weight = nx.shortest_path_length(simplified_G, nodes[index], nodes[index + 1], weight = 'length')
      except Exception:
        path.weight = 99999999
      exact_path_length += path.weight
      index = index + 1
  print("exact path length", exact_path_length)
  return guide_path

if __name__ == "__main__":
    G = ox.io.load_graphml('shandong_attr.graphml')
    simplified_G = ox.truncate.truncate_graph_bbox(G, north = 36.3, south = 34.9, east = 119.5, west = 116.8)
    origin = (119.324015, 35.157105)
    destination = (117.046638, 36.07307188)
    temp = [1, 1, 0]
    time_interval = [7058.0, 7587.0, 1885.0, 8703.0]
    # # with open('stay_region.pickle', "rb") as f:
    # #   all_region = pickle.load(f)
    # # all_embedding, stay_region = get_poi_embedding(5, all_region)
    # # with open('POI_embedding.pickle', 'wb') as f:
    # #   pickle.dump(all_embedding, f)
    # with open('POI_embedding.pickle', 'rb') as f:
    #   all_embedding = pickle.load(f)
    origin_node, origin_dist = ox.distance.nearest_nodes(simplified_G, origin[0], origin[1], return_dist = True)
    destination_node, destination_dist = ox.distance.nearest_nodes(simplified_G, destination[0], destination[1], return_dist = True)
    query = Query(popularity_threshold = 5, origin = origin, destination = destination, 
                  origin_node = origin_node, destination_node = destination_node, 
                  poi_seq = temp, poi_time_interval=time_interval)
    for i in range(len(all_embedding)):
      sub_embedding = []
      for path in all_embedding[i]:
        if path.end.popularity >= query.popularity_threshold:
          if 116.8 < path.end.cen_lng < 119.5:
            if 34.9 < path.end.cen_lat < 36.3:
              sub_embedding.append(path)
      embedding.append(sub_embedding)
    first_level = get_first_level(query, embedding)
    final_level = get_final_level(query, embedding)
    result_path = update_path(simplified_G, embedding, query, first_level, final_level)