
import numpy as np
import pandas
import os
import pickle
import math
import sparse
import numpy as np
from datetime import datetime

class Grid(object):
    def __init__(self, min_lon, min_lat, max_lon, max_lat, grid_res_lon_deg, grid_res_lat_deg):
        self.min_lon = min_lon
        self.min_lat = min_lat

        self.max_lon = max_lon
        self.max_lat = max_lat

        self.grid_res_lon_deg = grid_res_lon_deg
        self.grid_res_lat_deg = grid_res_lat_deg

        print(min_lon, max_lon, min_lat, max_lat)

        self.lon_size = (max_lon - min_lon + 1) / self.grid_res_lon_deg
        self.lat_size = (max_lat - min_lat + 1) / self.grid_res_lat_deg
 
        self.lon_tiles = math.ceil(self.lon_size / self.grid_res_lon_deg)
        self.lat_tiles = math.ceil(self.lat_size / self.grid_res_lat_deg)


def lat_lon_to_grid(grid, lat, lon):

    x = int((lon - grid.min_lon)/ grid.grid_res_lon_deg)
    y = int((lat - grid.min_lat) / grid.grid_res_lat_deg)
    return (x, y)


#Not requiring adjacency for now, as animals can cross larger distances in some cases
#   will grid based on stats on movement distances, etc.
 
def compute_transition_probabilities(movement_df, grid, checked = {}, total_count = {}, points = None):

    #Simplified to one action "move" for now - can add complexity later
    print((grid.lat_tiles, grid.lon_tiles, grid.lat_tiles, grid.lon_tiles, 1))
    if points is None: 
        points = {} #[[],[],[],[],[]]
    #grid.trans_prob = np.zeros((grid.lat_tiles, grid.lon_tiles, grid.lat_tiles, grid.lon_tiles, 1), dtype=np.float16)

    checked = {}
    total_count = {}
    for index, row in movement_df.iterrows():
      if row["id"] not in checked:
          checked[row["id"]] = True
          prev_state = lat_lon_to_grid(grid, row["lat"], row["lon"])
      else:
          current_state = lat_lon_to_grid(grid, row["lat"], row["lon"])
          if prev_state[0] not in points:
              points[prev_state[0]] = {prev_state[1] : {current_state[0] :\
                      {current_state[1] : {0 : 1}}}}
          elif prev_state[1] not in points[prev_state[0]]:
              points[prev_state[0]][prev_state[1]] = {current_state[0] :\
                                            {current_state[1] : {0 : 1}}}
          elif current_state[0] not in points[prev_state[0]][prev_state[1]]:
              points[prev_state[0]][prev_state[1]][current_state[0]] = {current_state[1] : {0 : 1}}
          elif current_state[1] not in points[prev_state[0]][prev_state[1]][current_state[0]]:
              points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]] = {0 : 1}
          else:
              points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]][0] += 1 

          if prev_state[0] not in total_count:
              total_count[prev_state[0]] = {prev_state[1] : 1}
          elif prev_state[1] not in total_count[prev_state[0]]:
              total_count[prev_state[0]][prev_state[1]] = 1
          else:
              total_count[prev_state[0]][prev_state[1]] += 1


    final_points = [[],[],[],[],[]]
    final_data = []
    for key in points:
        for key2 in points[key]:
            for key3 in points[key][key2]:
                for key4 in points[key][key2][key3]:
                    for key5 in points[key][key2][key3][key4]:
                        final_points[0].append(int(key))
                        final_points[1].append(int(key2))
                        final_points[2].append(int(key3))
                        final_points[3].append(int(key4))
                        final_points[4].append(int(key5))
                        final_data.append(points[key][key2][key3][key4][key5] / total_count[prev_state[0]][prev_state[1]])
    trans_prob = sparse.COO(final_points, final_data, shape=(grid.lat_tiles, grid.lon_tiles, grid.lat_tiles, grid.lon_tiles, 1))

    return checked, total_count, points, trans_prob



#For now, split - later, can investigate use of loss functions that account for missing data:
#https://arxiv.org/pdf/1911.06930
 
def split_movement_streams(movement_df, out_dir, run_uid):

     movement_df = movement_df.sort_values(by=['id', 'date'])
     dfs_by_id = [x for _, x in movement_df.groupby(movement_df["id"])]
     first_ind = 0
     final_dfs = {}
     for i in range(len(dfs_by_id)):
         for index, row in dfs_by_id[i].iterrows():
             if len(row["date"]) == 10:
                 row["date"] = row["date"] + " 00:00:00"
             if index < 1 or row["id"] not in final_dfs:
                 print(row["id"], final_dfs.keys())
                 last_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                 final_dfs[row["id"]] = []
                 continue
             current_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
             date_diff = current_date - last_date
             if int(round(float(date_diff.total_seconds() / (24.0*60*60)))) > 2:
                 final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:i])
                 first_ind = i
             last_date = current_date
         if first_ind < len(dfs_by_id[i]):
                 final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:])
         first_ind = 0
         
     for key in final_dfs:
         for i in range(len(final_dfs[key])):
             distance_diff = None
             print(key, len(final_dfs[key][i]))
             for index, row in final_dfs[key][i].iterrows():
                 if distance_diff is None:
                     distance_diff = []
                     last_coord = [row["lat"], row["lon"]]
                     continue
                 current_coord = [row["lat"], row["lon"]]
                 distance_diff.append(math.sqrt((current_coord[0] - last_coord[0])**2 + (current_coord[1] - last_coord[1])**2))
             print(min(distance_diff), max(distance_diff), np.mean(distance_diff), np.std(distance_diff))
             distance_diff = None
 
     with open(os.path.join(out_dir, run_uid + '_dfs.pkl'), 'wb') as f:
         pickle.dump(final_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

     return final_dfs



