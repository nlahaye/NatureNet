import requests
 
TOKEN = "ISVDirXgUjrJUbNDNzp8tOepMoBnN839Xe6bLEBCnUhbIitk8i6v5XoJxpXDuSkQE0f8DLvkTlK8UEIziAUTiw=="

req = requests.get('https://land.copernicus.eu/api/@datarequest_search?status=In_progress', headers={'Accept': 'application/json', 'Authorization': 'Bearer ' + TOKEN})
print(req.content)

