
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
  variables=["dos", "dos_error", "sea_ice_fraction", "sos", "sos_error"],
  minimum_longitude=-100,
  maximum_longitude=-60,
  minimum_latitude=20,
  maximum_latitude=40,
  start_datetime="2022-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=0,
  maximum_depth=0,
)




