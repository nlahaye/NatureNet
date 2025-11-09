
import numpy as np
import pandas
import os
import pickle

#TODO Grid object
class Grid(Object):
    def __init__(self, min_lon, min_lat, lon_size, lat_size, grid_res_lat_deg, grid_res_lon_deg):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.lat_size = lat_size
        self.lon_size = lon_size
 
        self.grid_res_lat_deg = grid_res_lat_deg
        self.grid_res_lon_deg = grid_res_lon_deg
        
        self.lat_tiles = lat_size / grid_res_lat_deg
        self.lon_tiles = lon_size / grid_res_lon_deg

        self.trans_prob = None


def lat_lon_to_grid(grid, lat, lon):

    x = int((lon - grid.min_lon)/ grid.grid_res_lon_deg)
    y = int((lat - grid.min_lat) / grid.grid_res_lat_deg)
    return (x, y)


#Not requiring adjacency for now, as animals can cross larger distances in some cases
#   will grid based on stats on movement distances, etc.
 
def compute_transition_probabilities(movement_df, grid):

    #Simplified to one action "move" for now
    grid.trans_prob = np.zeros((grid.lat_tiles, grid.lon_tiles, grid.lat_tiles, grid.lon_tiles, 1))

    checked = {}
    total_count = []
    for index, row in x.iterrows():
      if row["id"] not in checked:
          checked[row["id"]] = True
          prev_state = lat_long_to_grid(grid, row["lat"], row["lon"])
      else:
          current_state = lat_long_to_grid(grid, row["lat"], row["lon"])
          grid.trans_prob[prev_state[0], prev_state[1], current_state[0], current_state[1], 0] += 1
          if prev_state[0] not in total_count or prev_state[1] not in total_count[prev_state[0]]:
              total_count[prev_state[0]] = {prev_state[1] : 1}
          else:
              total_count[prev_state[0]][prev_state[1]] += 1

    #Normalize over all actions originating from previous state
    for key in total_count:
        for key2 in total_count[key]:
            grid.trans_prob[key, key2,:,:,:] = grid.trans_prob[key, key2,:,:,:] / total_count[key][key2]

    return grid


def split_movement_streams(movement_df):

     movement_df = movement_df.sort_values(by=['id', 'date'])
     dfs_by_id = [x for _, x in movement_df.groupby(movement_df["id"])]
     first_ind = 0
     final_dfs = {}
     for i in range(len(dfs_by_id)):
         for index, row in dfs_by_id[i].iterrows():
             if len(row["date"]) == 10:
                 row["date"] = row["date"] + " 00:00:00"
             if index < 1:
                 last_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                 final_dfs[row["id"]] = []
                 continue
             current_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
             date_diff = current_date - last_date
             if int(round((date_diff / 24*60*60))) > 2:
                 final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:i]
                 first_ind = i
             last_date = current_date
         if first_ind < len(dfs_by_id[i]):
                 final_dfs[row["id"]].append(dfs_by_id[i].iloc[first_ind:])
         first_ind = 0
         
     for key in final_dfs:
         for i in range(len(final_dfs[key]):
             print(key, len(final_dfs[key][i]))

     with open('hammerhead_dfs.pkl') as f:
         pickle.dump(final_dfs, f)
     




