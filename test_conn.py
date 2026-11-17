import http.client
conn = http.client.HTTPConnection('127.0.0.1', 8001, timeout=5)
conn.request('GET', '/')
r = conn.getresponse()
print(r.status, r.read()[:200].decode())
conn.close()