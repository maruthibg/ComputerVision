import psycopg2
from config import config


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        return conn


def get_cursor():
    """
    Get cursor of the connection
    """
    conn = connect()
    if not conn:
        raise ValueError('Postgres connection failed .....')
    cursor = conn.cursor()
    return cursor, conn


def select():
    try:
        cursor, conn = get_cursor()
        cursor.execute(
            "SELECT id, assetname, assetpath, status FROM Asset where status = 'To be Processed'")
        records = cursor.fetchall()
        records = (i for i in records)
        conn.close()
        return records
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def update(id):
    try:
        cursor, conn = get_cursor()
        cursor.execute(
            "UPDATE Asset set out = '%s' where id = '%s'" %
            (out, id))
        conn.commit
        conn.close()
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def insert():
    try:
        cursor, conn = get_cursor()
        cursor.execute("INSERT INTO Asset (id,assetname,assetpath,status,assetidentificationkey) \
        VALUES (1, 'laptop_1.mp4', 'D:\PROJECTS\maruthi_utils\scanner\videos', 'To be Processed', '')")
        conn.commit
        conn.close()
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
