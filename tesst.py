from pymilvus import connections, utility
connections.connect("default", uri="http://localhost:19530")
utility.drop_collection("database")

# from pymilvus import connections, utility

# # Đảm bảo URI_link là chính xác (localhost:19530)
# connections.connect("default", uri='http://localhost:19530')

# # Kiểm tra xem có collection nào tồn tại không
# print(utility.list_collections()) 

# # Thử kiểm tra collection 'database'
# print(utility.has_collection('database'))