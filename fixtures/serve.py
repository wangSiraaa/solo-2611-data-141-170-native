import argparse
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8148);a=p.parse_args()
s=ThreadingHTTPServer(('127.0.0.1',a.port),SimpleHTTPRequestHandler)
print('static fixture server port',s.server_port,flush=True);s.serve_forever()
