
import numpy as np
import pandas
import os
import pickle
import math
import sparse
import numpy as np
from datetime import datetime

N_ACTIONS = 9

class Grid(object):
    def __init__(self, min_lon, min_lat, max_lon, max_lat, grid_res_lon_deg, grid_res_lat_deg):
        self.min_lon = min_lon
        self.min_lat = min_lat

        self.max_lon = max_lon
        self.max_lat = max_lat

        self.grid_res_lon_deg = grid_res_lon_deg
        self.grid_res_lat_deg = grid_res_lat_deg

        self.lon_size = (max_lon - min_lon + 1) / self.grid_res_lon_deg
        self.lat_size = (max_lat - min_lat + 1) / self.grid_res_lat_deg
 
        self.lon_tiles = math.ceil(self.lon_size / self.grid_res_lon_deg)
        self.lat_tiles = math.ceil(self.lat_size / self.grid_res_lat_deg)


def grid_to_lat_lon(grid, x, y):

    lon = int(x)*grid.grid_res_lon_deg + grid.min_lon + (grid.grid_res_lon_deg / 2)
    lat = int(y)*grid.grid_res_lat_deg + grid.min_lat + (grid.grid_res_lat_deg / 2)

    return (lon, lat)

def lat_lon_to_grid(grid, lat, lon):

    x = int((lon - grid.min_lon)/ grid.grid_res_lon_deg)
    y = int((lat - grid.min_lat) / grid.grid_res_lat_deg)
    return (x, y)

def grid_to_ind(grid, x, y):
    return (y*grid.lon_tiles) + x

#Not requiring adjacency for now, as animals can cross larger distances in some cases
#   will grid based on stats on movement distances, etc.
 
def compute_transition_probabilities(movement_df, grid, total_count = {}, points = None):

    if points is None: 
        points = {} #[[],[],[],[],[]]
    #grid.trans_prob = np.zeros((grid.lat_tiles, grid.lon_tiles, grid.lat_tiles, grid.lon_tiles, 1), dtype=np.float16)

    deg_grid = np.zeros((grid.lat_tiles, grid.lon_tiles, 2))
    for lt in range(grid.lat_tiles):
        for ln in range(grid.lon_tiles):
            glon, glat = grid_to_lat_lon(grid, ln, lt)
            deg_grid[lt,ln,0] = glat
            deg_grid[lt,ln,1] = glon

    total_count = {}
    actions = []
    single_ind_positions = []
    distance = []
    prev_state = None
    act_ind = 0
    for index, row in movement_df.iterrows():
        if act_ind == 0:
          prev_state = lat_lon_to_grid(grid, row["lat"], row["lon"])
          current_state = lat_lon_to_grid(grid, row["lat"], row["lon"])
        else:
          current_state = lat_lon_to_grid(grid, row["lat"], row["lon"])

          #TODO - there are a few exceptions to these simplified rule, given that a map could wrap the globe, but keeping simple for now
          # action 0 = stay. Other 8 actions are cardinal + diagonal directions in clockwise order, starting with NW diagonal movement == 1
          action = -1
          if current_state[0] == prev_state[0]:
             if current_state[1] == prev_state[1]: 
                 action = 0 
             elif current_state[1] > prev_state[1]: 
                 action = 8
             else:
                 action = 4
          elif current_state[0] < prev_state[0]:
              if current_state[1] == prev_state[1]:
                  action = 2
              elif current_state[1] > prev_state[1]:
                  action = 3
              else:
                  action = 4
          else:
              if current_state[1] == prev_state[1]:
                  action = 6
              elif current_state[1] > prev_state[1]:
                  action = 5
              else:
                  action = 7
          actions.append(action)

          if prev_state[0] not in points:
              points[prev_state[0]] = {prev_state[1] : {current_state[0] :\
                      {current_state[1] : {action : 1}}}}
          elif prev_state[1] not in points[prev_state[0]]:
              points[prev_state[0]][prev_state[1]] = {current_state[0] :\
                                            {current_state[1] : {action : 1}}}
          elif current_state[0] not in points[prev_state[0]][prev_state[1]]:
              points[prev_state[0]][prev_state[1]][current_state[0]] = {current_state[1] : {action : 1}}
          elif current_state[1] not in points[prev_state[0]][prev_state[1]][current_state[0]]:
              points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]] = {action : 1}
          elif action not in points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]]:
              points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]][action] = 1
          else:
              points[prev_state[0]][prev_state[1]][current_state[0]][current_state[1]][action] += 1 

          if prev_state[0] not in total_count:
              total_count[prev_state[0]] = {prev_state[1] : 1}
          elif prev_state[1] not in total_count[prev_state[0]]:
              total_count[prev_state[0]][prev_state[1]] = 1
          else:
              total_count[prev_state[0]][prev_state[1]] += 1

        single_ind_positions.append(grid_to_ind(grid, current_state[0], current_state[1]))
        distance_grid = np.zeros((grid.lat_tiles, grid.lon_tiles))
        lon, lat = grid_to_lat_lon(grid, current_state[1], current_state[0])
        for lt in range(grid.lat_tiles):
            for ln in range(grid.lon_tiles):
                distance_grid[lt,ln] = math.sqrt((lat - deg_grid[lt,ln,0])**2 + (lon - deg_grid[lt,ln,1])**2)
        distance.append(distance_grid)
        act_ind = act_ind + 1 

    #final_points = [[],[],[],[],[]]
    final_points = [[],[],[]]
    final_data = []
    for key in points:
        for key2 in points[key]:
            for key3 in points[key][key2]:
                for key4 in points[key][key2][key3]:
                    for key5 in points[key][key2][key3][key4]:
                        flat_ind = grid_to_ind(grid, int(key), int(key2)) 
                        #SWIRL works with flat index structure currentl 
                        #TODO investigate tradeoffs of moving back to lat/lon map. I suspect this is correlated to MLP prediction of positions (easier for flat, single number)
                        #May just be interpretability / usability ?
                        flat_ind_2 = grid_to_ind(grid, int(key3), int(key4)) 
                        #final_points[0].append(int(key))
                        #final_points[1].append(int(key2))
                        #final_points[2].append(int(key3))
                        #final_points[3].append(int(key4))
                        #final_points[4].append(int(key5))
                        final_points[0].append(flat_ind) #initial position
                        final_points[1].append(int(key5)) #action
                        final_points[2].append(flat_ind_2) #final position
                        
                        final_data.append(points[key][key2][key3][key4][key5] / total_count[prev_state[0]][prev_state[1]])
                        n_states = grid.lon_tiles * grid.lat_tiles

    trans_prob = sparse.COO(final_points, final_data, shape=(n_states, N_ACTIONS, n_states))

    return total_count, points, trans_prob, actions, single_ind_positions, distance



#For now, split - later, can investigate use of loss functions that account for missing data:
#https://arxiv.org/pdf/1911.06930
 
def split_movement_streams(movement_df, out_dir, run_uid):

     movement_df = movement_df.sort_values(by=['id', 'date'])
     dfs_by_id = [x for _, x in movement_df.groupby(movement_df["id"])]
     #first_ind = 0
     final_dfs = {}
     for i in range(len(dfs_by_id)):
         first_ind = 0
         actual_ind = -1
         for index, row in dfs_by_id[i].iterrows():
             actual_ind = actual_ind +1
             if len(row["date"]) == 10:
                 row["date"] = row["date"] + " 00:00:00"
             if actual_ind < 1 or row["id"] not in final_dfs:
                 actual_ind = 0
                 first_ind = 0
                 last_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                 final_dfs[row["id"]] = []
                 continue
             current_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
             date_diff = current_date - last_date
             if int(round(float(date_diff.total_seconds() / (24.0*60*60)))) > 2 or (actual_ind-first_ind +1) >= 100 or actual_ind == len(dfs_by_id[i])-1:
                 final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:actual_ind])
                 first_ind = actual_ind
             last_date = current_date
         #if first_ind < len(dfs_by_id[i]):
         #        final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:])
         first_ind = 0
      
     for key in final_dfs:
         for i in range(len(final_dfs[key])):
             distance_diff = None
             for index, row in final_dfs[key][i].iterrows():
                 if distance_diff is None:
                     distance_diff = []
                     last_coord = [row["lat"], row["lon"]]
                     continue
                 current_coord = [row["lat"], row["lon"]]
                 distance_diff.append(math.sqrt((current_coord[0] - last_coord[0])**2 + (current_coord[1] - last_coord[1])**2))
             distance_diff = None
 
     with open(os.path.join(out_dir, run_uid + '_dfs.pkl'), 'wb') as f:
         pickle.dump(final_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

     return final_dfs



