from sqlalchemy import create_engine  
from sqlalchemy import Column, String  
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker

from utils import Packet

db_string = "postgres://postgres:Floorcheck@1234@216.10.249.58:5432/floor_inv"
#db_string = "postgresql://dev61:pass123@dafslx20/nagel_brijith"

db = create_engine(db_string)  
base = declarative_base()

class Ledger(base):  
    __tablename__ = 'Asset'

    assetid = Column(String, primary_key=True)
    assetname = Column(String)
    assetpath = Column(String)
    assetstatus = Column(String)
    assetidentificationkey = Column(String)

Session = sessionmaker(db)  
session = Session()

base.metadata.create_all(db)

# Create 
# ledgers = Ledger(assetid='2', assetname="Doctor Strange", assetpath="Scott Derrickson", assetstatus="2016")  
# session.add(ledgers)  
# session.commit()

def get_assets(status):
    results = []
    rows = session.query(Ledger)
    if status:
        rows = rows.filter(Ledger.assetstatus == status)
    for row in rows:
        p = Packet()
        p.id = row.assetid
        p.name = row.assetname
        p.path = row.assetpath
        p.status = row.assetstatus
        results.append(p)
    return results

def update(assetid, key, status):
    # Update
    ledger = session.query(Ledger).filter_by(Ledger.assetid==assetid).first()
    if ledger:
        value = [i.assetidentificationkey for i in ledger][0]
        ledger.assetidentificationkey = '%s,%s'(value, key)
        ledger.assetstatus = status
        session.commit()
    
def failure(assetid, status='Failure'):
    # Update
    ledger = session.query(Ledger).filter_by(Ledger.assetid==assetid).first()
    ledger.assetstatus = status
    session.commit()
