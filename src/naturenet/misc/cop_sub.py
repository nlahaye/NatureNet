import copernicusmarine

vrs = [
"cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m",
"cmems_mod_glo_bgc_anfc_0.25deg_static",
"cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i",
"cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i",
"cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
"cmems_mod_glo_phy_anfc_0.083deg_static",
"cmems_mod_glo_phy_anfc_0.083deg_static",
"cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m",
"cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
"cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m",
"cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
"cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
"cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
"cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
] 


for vr in vrs:
    try: 
        copernicusmarine.subset(
          dataset_id=vr,
          #variables=["uo", "vo"],
          minimum_longitude=-126,
          maximum_longitude=-102,
          minimum_latitude=13,
          maximum_latitude=40,
          start_datetime="2017-07-10",
          end_datetime="2017-12-11",
          #minimum_depth=0,
          #maximum_depth=30,
          output_filename = vr + ".nc",
          output_directory = "copernicus-data-whales"
        )
    except:
        pass

 

