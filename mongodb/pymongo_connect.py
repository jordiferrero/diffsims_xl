from pymongo import MongoClient
from secrets import mongo_db_connection_string

def connect_to_mongo_database(database_str_name, collection_str_name):

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = mongo_db_connection_string

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    dbname = client[database_str_name]

    # Create a collection
    return dbname[collection_str_name]
