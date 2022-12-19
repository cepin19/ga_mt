import sys
import requests
for line in sys.stdin:
    r=requests.post("http://localhost:5233/search/1/en/cz/",  json=[(line)])
    print(r.json())

