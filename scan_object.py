import os
import subprocess
import argparse

from database import get_fail_assets, update, failure

def main(cwd, executable, script, path, enable_imshow, ooi):
    os.chdir(r'%s'%cwd)
    ping = subprocess.Popen(["%s %s -p %s -e %s"%(executable, script, path, enable_imshow)],stdout = subprocess.PIPE,stderr = subprocess.PIPE,shell=True)
    out = ping.communicate()[0]
    output = str(out, 'utf-8')
    print(output)

def process(cwd, executable, script, enable_imshow=0, ooi=0):
    print('Processing from database ....')
    print('Starts at : %s'%str(datetime.datetime.now()))
    assets = get_fail_assets()
    if assets:
        for asset in assets:
            video = os.path.join(asset.path, asset.name)
            if video.endswith('.pdf'):
                continue                
            if (not video.endswith('.md')):
                print('Processing video file - %s'%(video))
                string = main(cwd, executable, script, video, enable_imshow, ooi)
                if string:
                    #update(asset.id, string)
                    print(string)
                    return string
                else:
                    print('Failed 1')
                    failure(asset.id)
            else:
                print('Failed 2')
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
