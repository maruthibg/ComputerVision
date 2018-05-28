import psycopg2
from config import config
from utils import Packet


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


def get_assets():
    results = []
    cursor, conn = get_cursor()
    cursor.execute(
        "SELECT assetid, assetname, assetpath, assetstatus FROM Asset where \
        assetstatus = 'To be Processed'")
    records = cursor.fetchall()
    
    for record in records:
        p = Packet()
        p.id = record[0]
        p.name = record[1].strip()
        p.path = record[2].strip()
        results.append(p)
    cursor.close()
    conn.close()
    return results

def update(id, value, status='Processed'):
    cursor, conn = get_cursor()
    statement = """UPDATE Asset set assetidentificationkey='%s', assetstatus='%s' where assetid='%s'""" %(value, status, id)
    cursor.execute(statement)
    print(statement)    
    conn.commit()
    cursor.close()
    conn.close()

def failure(key):
    cursor, conn = get_cursor()
    statement = """UPDATE Asset set assetstatus='%s' where assetid='%s'""" %('Failure', key)
    print(statement)
    cursor.execute(statement)
    conn.commit()
    cursor.close()
    conn.close()
