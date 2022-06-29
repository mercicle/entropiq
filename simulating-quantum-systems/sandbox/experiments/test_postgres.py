
from sqlalchemy import create_engine
import requests
import psycopg2

def load_environ(environ_path):

    with open(environ_path) as f:
        
        for line in f:
            if ('=' not in line) or line.startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            os.environ[key] = value
         
load_environ(os.getcwd()+"/db_creds.env")
db_string = "postgresql://" + \
                    os.getenv("POSTGRES_DB_USERNAME") + \
                    ":" + \
                    os.getenv("POSTGRES_DB_PASSWORD") + \
                    "@" + \
                    os.getenv("POSTGRES_DB_URL") + \
                    ":" + \
                    os.getenv("POSTGRES_DB_PORT") + \
                    "/" + \
                    os.getenv("POSTGRES_DB_NAME")

engine = create_engine(db_string)
connection = engine.connect()
