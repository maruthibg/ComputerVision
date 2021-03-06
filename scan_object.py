import os
import datetime
import subprocess
import argparse

#from database import get_assets, update, failure
from orm import get_assets, update, failure

def main(cwd, executable, script, path, enable_imshow, ooi):
    os.chdir(r'%s'%cwd)
    ping = subprocess.Popen(["%s %s -p %s -e %s"%(executable, script, path, enable_imshow)],stdout = subprocess.PIPE,stderr = subprocess.PIPE,shell=True)
    out = ping.communicate()[0]
    output = str(out, 'utf-8')
    return output

def process(cwd, executable, script, enable_imshow=0, ooi=0):
    print('Processing from database ....')
    print('Starts at : %s'%str(datetime.datetime.now()))
    status = 'To be Processed'
    assets = get_assets(status=status)
    if assets:
        for asset in assets:
            path = os.path.join(asset.path, asset.name)
            if path.endswith('.pdf') or path.endswith('.md'):
                continue
            print('Processing video file - %s'%(path))
            string = main(cwd, executable, script, path, enable_imshow, ooi)
            print(string)
            if string:
                update(asset.id, string.strip(), assetidentificationkey2=True, status='Processed Stage 1')
                print(string)
            else:
                print('Failed 1')
                failure(asset.id)
    print('Ends at : %s'%str(datetime.datetime.now()))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="Current Working Directory")
    ap.add_argument("-e", "--executable", required=True, help="Excutable script with path")
    ap.add_argument("-s", "--script", required=True, help="Excutable script with path")
    ap.add_argument("-p", "--path", required=False, help="Path to input video")
    ap.add_argument("-i", "--imshow", required=False, help="Enable Imshow")
    ap.add_argument("-o", "--ooi", required=False, help="Object of interest")
    
    args = vars(ap.parse_args())
    cwd = args["directory"]
    executable = args["executable"]
    script = args["script"]
    path = args["path"]
    enable_imshow = int(args.get("imshow", '0'))
    ooi = args["ooi"]

    process(cwd, executable, script, enable_imshow, ooi)
