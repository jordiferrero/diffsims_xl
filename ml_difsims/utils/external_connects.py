from paramiko.client import AutoAddPolicy
from pymongo import MongoClient
from paramiko import SSHClient

import secrets

def connect_to_mongo_database(database_str_name, collection_str_name):

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = secrets.mongo_db_connection_string

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    dbname = client[database_str_name]

    # Create a collection
    return dbname[collection_str_name]


def start_shh_connection():
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(hostname='rds.uis.cam.ac.uk', port=22, username=secrets.RAVEN_USERNAME, password=secrets.RAVEN_PASSWORD)
    return ssh
