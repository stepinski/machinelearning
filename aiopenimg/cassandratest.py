from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from threading import Event
from cassandra.auth  import PlainTextAuthProvider
import time
import sys

class PagedResultHandler(object):

    def __init__(self, future):
        self.error = None
        self.finished_event = Event()
        self.future = future
        self.future.add_callbacks(
            callback=self.handle_page,
            errback=self.handle_error)

    def handle_page(self, rows):
        #for row in rows:
            #process_row(row)

        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
        else:
            self.finished_event.set()
    def handle_error(self, exc):
        self.error = exc
        self.finished_event.set()

def process_row(user_row):
    print user_row.channel, user_row.timestamp, user_row.value

channel = sys.argv[1]
auth_provider = PlainTextAuthProvider(
        username='admin', password='admin')

cluster = Cluster(['10.150.1.69'],auth_provider=auth_provider)
session = cluster.connect()

start = time.time()
query = "SELECT * FROM production.data where channel='{}'".format(channel)
statement = SimpleStatement(query, fetch_size=10000)

future = session.execute_async(statement)
handler = PagedResultHandler(future)
handler.finished_event.wait()
if handler.error:
    raise handler.error
end = time.time()
print ( end - start)
cluster.shutdown()

