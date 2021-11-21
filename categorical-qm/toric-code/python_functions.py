
from pymongo import MongoClient
import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

def get_mongo_db(username = os.getenv("STARSTUFF_MONGODB_USERNAME"), password = os.getenv("STARSTUFF_MONGODB_PASSWORD"), clustername = os.getenv("STARSTUFF_MONGODB_CLUSTERNAME"), db_name = os.getenv("STARSTUFF_MONGODB_DBNAME")):
    connection_string = "mongodb+srv://"+username+":"+password+"@"+clustername+".mongodb.net/myFirstDatabase"
    client = MongoClient(connection_string)
    return client[db_name]

if __name__ == "__main__":

    # Get the database
    dbname = get_mongo_db()
    print(dbname)
