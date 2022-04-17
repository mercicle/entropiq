import os, sys
import pandas as pd
import requests
import psycopg2
from sqlalchemy import create_engine

def load_environ(environ_path):
    with open(environ_path) as f:
        for line in f:
            if ('=' not in line) or line.startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

#load_environ(os.getcwd()+"/db_creds.env")

def get_postgres_conn():
    conn_string = "postgresql://" + \
                os.getenv("POSTGRES_DB_USERNAME") + \
                ":" + \
                os.getenv("POSTGRES_DB_PASSWORD") + \
                "@" + \
                os.getenv("POSTGRES_DB_URL") + \
                ":" + \
                os.getenv("POSTGRES_DB_PORT") + \
                "/" + \
                os.getenv("POSTGRES_DB_NAME")
    engine = create_engine(conn_string)
    try:
       conn = engine.connect()
       results = connection.execute(sql_string_get_version).fetchone()
       print("Connected to version: {}".format(results[0]))
    except:
        conn = None
        print("Unable to establish a connection ¯\\_(ツ)_//¯")
    return conn

def write_table(conn, df, table_name, schema_name, chunk_size=int(1e3), append=False):
    add_or_replace = lambda x: 'append' if append else 'replace'
    try:
        df.to_sql(name=table_name, con=conn, schema=schema_name, method='multi', if_exists = add_or_replace(append), index=False, index_label=None, chunksize=chunk_size)
    except:
        print("Failed to save the data to " + schema_name + "." + table_name + "  ¯\\_(ツ)_//¯.")

def get_table(conn, table_name, schema_name, where_string="", only_these_cols=[], use_table_quote=False):

    if(len(only_these_cols)>0):
        select_col_string = ','.join(only_these_cols)
    else:
        select_col_string = " * "

    if use_table_quote:
        query_string = "select " + select_col_string + " from " + schema_name + """.\"""" + table_name + """\"""" + where_string
    else:
        query_string = "select " + select_col_string + " from " + schema_name + "." + table_name + " " + where_string

    print("SQL Statement: " + query_string)
    df = pd.read_sql(query_string, con = conn)

    return df
