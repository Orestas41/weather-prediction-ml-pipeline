conn = http.client.HTTPSConnection("archive-api.open-meteo.com")
payload = ''
headers = {}
conn.request("GET", f"/v1/archive?latitude=51.45&longitude=-2.58&start_date=2023-05-18&end_date=2023-10-20&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
res = conn.getresponse()
data = res.read()
data = json.loads(data.decode("utf-8"))
with open(f"./data/data.json", "w") as write_file:
    json.dump(data, write_file)