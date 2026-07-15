import requests
import datetime

 
TOKEN_ID = "dfcd7383be57ed02bf2ab0c7d663c94bd1dd05a1"

URL = 'https://land.copernicus.eu/api/@datarequest_post'

TOKEN = "_1I8gcaoKqpdIRIIymGoJ5HxchcEOx3aflVkNs564PYXVswPSkV1PIV5MjSzVGtmbcNd_RpoH2CpZSYclSOBXg=="

HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': "Bearer " + TOKEN}


sdate = datetime.datetime.strptime("2021-08-31T00:00:00", "%Y-%m-%dT%H:%M:%S")
edate = datetime.datetime.strptime("2021-09-06T23:59:59", "%Y-%m-%dT%H:%M:%S")

tm_int = 0

total_end = datetime.datetime.strptime("2021-11-02T00:00:00", "%Y-%m-%dT%H:%M:%S")

while sdate < total_end:
    if tm_int > 0:
        sdate = sdate + datetime.timedelta(days=6)
        edate = edate + datetime.timedelta(days=6)

    sseconds = int(round(sdate.timestamp() * 1000))
    eseconds = int(round(edate.timestamp() * 1000))

    print(sseconds, eseconds)

    JSON = {'Datasets': [
    #"@id": "https://land.copernicus.eu/api/en/products/clc-backbone/clc-backbone-2021",
    #{'DatasetID': '4d0d78ad472c45819aff1d9fa7af0461', 
    #'DatasetDownloadInformationID': 'b9461c94-2e4e-4058-81c4-b274c0e8b12b', 
    #'NUTS': 'AT', 
    #'OutputFormat': 'Geotiff', 
    #'OutputGCS': 'EPSG:4326'},
    #END DATE 1635724800000
    #tree-cover-density-2021

    #{'DatasetID': "508fbc982b11474aba8ea00a8a529ea3",
    #'DatasetDownloadInformationID': "3e14631c-e1d6-45b7-adac-bfc970032c1b",
    #'NUTS': 'AT',
    #'OutputFormat': 'Geotiff',
    #'OutputGCS': 'EPSG:4326'},
    #'TemporalFilter': {'StartDate': sseconds, 'EndDate': eseconds}},


    #10-daily-soil-water-index-global-12-5-km-v4
    {'DatasetID': "aa72dde6057a4ababfa0ffac960d0988",
    'DatasetDownloadInformationID': "8ea4e7f0-06a6-4527-9850-b103226333ad",
    'NUTS': 'AT',
    'OutputFormat': 'Geotiff',
    'OutputGCS': 'EPSG:4326',
    'TemporalFilter': {'StartDate': sseconds, 'EndDate': eseconds}},

    #high-resolution-gap-filled-fractional-snow-cover
    {'DatasetID': "8d14c6a2ba7946faaaf50e4e8241fe1c",
    'DatasetDownloadInformationID': "2c51dd60-0e70-4ff8-88c0-bd8cb6c58123",
    'NUTS': 'AT',
    'OutputFormat': 'Geotiff',
    'OutputGCS': 'EPSG:4326',
    'TemporalFilter': {'StartDate': sseconds, 'EndDate': eseconds}},

    #normalised-difference-vegetation-index-v2-0-300m
    {'DatasetID': "5bc2feef2e27456792b3d2303d53ded5",
    'DatasetDownloadInformationID': "e4662555-eb53-4e45-a3d2-45f6eb044d85",
    'NUTS': 'AT',
    'OutputFormat': 'Geotiff',
    'OutputGCS': 'EPSG:4326',
    'TemporalFilter': {'StartDate': sseconds, 'EndDate': eseconds}},

    #burnt-area-v3-1-monthly-300m
    {'DatasetID': "15353abe103e4be0aa5c9b82cee5a4d5",
    'DatasetDownloadInformationID': "e38378ca-6fea-40a9-8939-a5425a4bdf91",
    'NUTS': 'AT',
    'OutputFormat': 'Geotiff',
    'OutputGCS': 'EPSG:4326',
    'TemporalFilter': {'StartDate': sseconds, 'EndDate': eseconds}}


    ]
    }

    response = requests.post(URL, headers=HEADERS, json=JSON)

    print(response.content)

    tm_int = tm_int + 1

#requests.post('https://land.copernicus.eu/api/@datarequest_post', headers={'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer <REDACTED>'}, json={'Datasets': [{'DatasetID': 'a8d945f0edd143a0a5240c28bafa23da', 'DatasetDownloadInformationID': 'f328dbcc-9069-4240-b8bb-6d5df918671a', 'NUTS': 'EC', 'OutputFormat': 'Geotiff', 'OutputGCS': 'EPSG:4326'}]})

#requests.post('https://land.copernicus.eu/api/@datarequest_post', headers={'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer <REDACTED>'}, json={'Datasets': [{'DatasetID': 'a8d945f0edd143a0a5240c28bafa23da', 'DatasetDownloadInformationID': 'f328dbcc-9069-4240-b8bb-6d5df918671a', 'NUTS': 'ES522', 'OutputFormat': 'Geotiff', 'OutputGCS': 'EPSG:4326'}]})

#requests.post('https://land.copernicus.eu/api/@datarequest_post', headers={'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer <REDACTED>'}, json={'Datasets': [{'DatasetID': '7caf242841a34d44a2818f683b111ea9', 'DatasetDownloadInformationID': 'e5bd1ea3-0873-4abf-99ca-3a487aba0fc0', 'OutputFormat': 'Geotiff', 'OutputGCS': 'EPSG:4326', 'BoundingBox': [2.354736328128108, 46.852958688910306, 4.639892578127501, 45.88264619696234], 'TemporalFilter': {'StartDate': 1546333200000, 'EndDate': 1559289600000}}]})

#requests.post('https://land.copernicus.eu/api/@datarequest_post', headers={'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer <REDACTED>'}, json={'Datasets': [{'DatasetID': '7caf242841a34d44a2818f683b111ea9', 'DatasetDownloadInformationID': 'e5bd1ea3-0873-4abf-99ca-3a487aba0fc0', 'OutputFormat': 'Geotiff', 'OutputGCS': 'EPSG:4326', 'BoundingBox': [2.354736328128108, 46.852958688910306, 4.639892578127501, 45.88264619696234], 'TemporalFilter': {'StartDate': 1546333200000, 'EndDate': 1559289600000}}]})

 
#requests.post('https://land.copernicus.eu/api/@datarequest_post', headers={'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer <REDACTED>'}, json={'Datasets': [{'DatasetID': 'a8d945f0edd143a0a5240c28bafa23da', 'DatasetDownloadInformationID': 'f328dbcc-9069-4240-b8bb-6d5df918671a', 'NUTS': 'EC', 'OutputFormat': 'Geotiff', 'OutputGCS': 'EPSG:4326'}]})

