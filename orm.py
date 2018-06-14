from sqlalchemy import create_engine  
from sqlalchemy import Column, String  
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker

from utils import Packet

db_string = "postgres://postgres:Floorcheck@1234@216.10.249.58:5432/floor_inv"

db = create_engine(db_string)  
base = declarative_base()

class Assets(base):  
    __tablename__ = 'asset'

    assetid = Column(String, primary_key=True)
    assetname = Column(String)
    assetpath = Column(String)
    assetstatus = Column(String)
    assetidentificationkey = Column(String)

Session = sessionmaker(db)  
session = Session()

#base.metadata.create_all(db)

# Create 
# ledgers = Ledger(assetid='2', assetname="Doctor Strange", assetpath="Scott Derrickson", assetstatus="2016")  
# session.add(ledgers)  
# session.commit()

def get_assets(status=False):
    results = []
    rows = session.query(Assets)
    if status:
        rows = rows.filter(Assets.assetstatus == status)
    for row in rows:
        p = Packet()
        p.id = row.assetid
        p.name = row.assetname
        p.path = row.assetpath
        p.status = row.assetstatus
        results.append(p)
    return results

def update(assetid, key, status=''):
    # Update
    asset = session.query(Assets).filter(Assets.assetid==assetid).first()
    if asset:
        value = [i.assetidentificationkey for i in asset][0]
        asset.assetidentificationkey = '%s,%s'(value, key)
        if status:
            asset.assetstatus = status
        session.commit()
    
def failure(assetid, status='Failure'):
    # Update
    asset = session.query(Assets).filter(Assets.assetid==assetid).first()
    asset.assetstatus = status
    session.commit()
    
if __name__ == '__main__':
    print(get_assets())
