import datetime
import requests
import json
import jwt
import time


now = int(time.time()) - 3000

# Load saved key from filesystem
service_key = json.load(open('token_2.json', 'rb'))

private_key = service_key['private_key'].encode('utf-8')
print(private_key)

print(service_key['user_id'], service_key['token_uri'], service_key['client_id'])
print(datetime.datetime.utcnow()) #int(time.time()), int(time.time() + (60 * 60)))

claim_set = {
    "iss": service_key['client_id'],
    "sub": service_key['user_id'],
    "aud": service_key['token_uri'],
    "iat": now - 60,
    "exp": now + 3600 - 60 #int(time.time() + (60 * 60)),
}
grant = jwt.encode(claim_set, private_key, algorithm='RS256')

print(claim_set)

print(grant)

#print(datetime.datetime.utcnow(), jwt.decode(grant, algorithm='RS256'))

response = requests.post('https://land.copernicus.eu//@@oauth2-token', headers={'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}, data={'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer', 'assertion': grant})

print(response.content)
 
