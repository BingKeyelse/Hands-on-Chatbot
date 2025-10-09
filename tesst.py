from pymilvus import connections, utility
connections.connect("default", uri="http://localhost:19530")
utility.drop_collection("database")