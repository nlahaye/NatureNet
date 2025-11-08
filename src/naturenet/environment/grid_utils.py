

from astropy_healpix import HEALPix
from astropy import units as u


#TODO Grid object

def lat_lon_to_grid(grid, lat, lon):

    x = int((lon - grid.min_lon)/ grid.lon_size)
    y = int((lat - grid.min_lat) / grid.lat_size)
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





