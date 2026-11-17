import http.client
import time

# Wait a moment for server to start
time.sleep(2)

conn = http.client.HTTPConnection('127.0.0.1', 8001, timeout=5)
conn.request('GET', '/')
r = conn.getresponse()
print(f"Status: {r.status}")
print(f"Response head: {r.read()[:300].decode()}")
conn.close()