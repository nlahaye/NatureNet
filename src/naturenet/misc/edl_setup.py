from urllib import request, parse
from http.cookiejar import CookieJar
from base64 import b64encode
import getpass
import netrc
import requests
import json
import os
import pprint
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio
from rasterio.plot import show
import numpy as np
import time
from netCDF4 import Dataset
from owslib.wms import WebMapService
from owslib.util import Authentication



def setup_earthdata_login_auth(endpoint):
    """
    Set up the request library so that it authenticates against the given Earthdata Login
    endpoint and is able to track cookies between requests.  This looks in the .netrc file
    first and if no credentials are found, it prompts for them.

    Valid endpoints include:
        uat.urs.earthdata.nasa.gov - Earthdata Login UAT (Harmony's current default)
        urs.earthdata.nasa.gov - Earthdata Login production
    """
    token = None
    try:
        username, _, password = netrc.netrc().authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        print('Please provide your Earthdata Login credentials to allow data access')
        print('Your credentials will only be passed to %s and will not be exposed in Jupyter' % (endpoint))
        username = input('Username:')
        print('Password:')
        password = getpass.getpass()

    # Retrieve existing bearer token or generate a new one if none exists
    basic_auth_string = f'{username}:{password}'
    basic_auth_bytes = basic_auth_string.encode('ascii')
    base64_bytes = b64encode(basic_auth_bytes)
    base64_basic_auth = base64_bytes.decode('ascii')
    try:
        token_response = requests.get(f'https://{endpoint}/api/users/tokens', headers={'Authorization': f'Basic {base64_basic_auth}'})
        token_response_json = token_response.json()
        if token_response_json:
            token = token_response_json[0]['access_token']
        else:
            token_response = requests.post(f'https://{endpoint}/api/users/token', headers={'Authorization': f'Basic {base64_basic_auth}'})
            token_response_json = token_response.json()
            token = token_response_json['access_token']
    except:
        print(f'Error: token response: {token_response_json}')

    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)

    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)
    return token

class BearerToken(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers['Authorization'] = f'Bearer {self.token}'
        return r


bearer_token = setup_earthdata_login_auth('uat.urs.earthdata.nasa.gov')

def submit_request(url):
    status_req = request.Request(url)
    status_req.add_header('Authorization', f'Bearer {bearer_token}')

    # Open the request and read the response
    with request.urlopen(status_req) as response:
        return response.read()

def get_json_response(url):
    response_data = submit_request(url)
    return json.loads(response_data)

short_name = "SWOT_L2_LR_SSH_WINDWAVE_D"
version = "D"


params1 = params = {
    'short_name': short_name,
    'version': version}

cmr_collections_url = 'https://cmr.uat.earthdata.nasa.gov/search/collections.json'
query_string = parse.urlencode(params)
url = cmr_collections_url + "?" + query_string
print(url)
cmr_response = request.urlopen(url)
cmr_results = json.loads(cmr_response.read().decode('utf-8'))
print(cmr_results)
collectionlist = [el['id'] for el in cmr_results['feed']['entry']]
collection_id = collectionlist[0]
print(collection_id)

capabilities_url = f'https://harmony.uat.earthdata.nasa.gov/capabilities?collectionId={harmony_collection_id}'
capabilities = get_json_response(capabilities_url)
capabilities['services']


harmony_root = 'https://harmony.uat.earthdata.nasa.gov'

params = {
    'short_name': short_name,
    'version': version,
    'time': '("2023-09-01T00:00:00.000Z":"2024-10-07T03:00:00.000Z")',
    'variable' : 'mean_wave_direction,mean_wave_period_t02,swh_karin_2,wind_speed_karin_2',
    'lat': '(20:40)',
    'lon': '(-100:-60)',
    'outputCrs': 'EPSG:4326',
    'format': 'image/tiff'
}

x_url = harmony_root+'/{collection_id}/ogc-api-coverages/{ogc-api-coverages_version}/collections/{variable}/coverage/rangeset?granuleid={granuleid}&subset=lat{lat}&subset=lon{lon}&subset=time{time}&outputCrs={outputCrs}&format={format}'.format(**params)
 
print('Request URL', x_url)
time_results = submit_request(x_url)

time_file_name = 'harmonytimesubset.tif'
time_filepath = str(time_file_name)
file_ = open(time_filepath, 'wb')
file_.write(time_results)
file_.close()



