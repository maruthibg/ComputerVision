import os
import subprocess
import argparse

def main(cwd, executable, script, path, enable_imshow, ooi):
    os.chdir(r'%s'%cwd)
    ping = subprocess.Popen(["%s %s -p %s -e %s"%(executable, script, path, enable_imshow)],stdout = subprocess.PIPE,stderr = subprocess.PIPE,shell=True)
    out = ping.communicate()[0]
    output = str(out)
    print(output)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="Current Working Directory")
    ap.add_argument("-e", "--executable", required=True, help="Excutable script with path")
    ap.add_argument("-s", "--script", required=True, help="Excutable script with path")
    ap.add_argument("-p", "--path", required=True, help="Path to input video")
    ap.add_argument("-i", "--imshow", required=True, help="Enable Imshow")
    ap.add_argument("-o", "--ooi", required=False, help="Object of interest")
    
    args = vars(ap.parse_args())
    cwd = args["directory"]
    executable = args["executable"]
    script = args["script"]
    path = args["path"]
    enable_imshow = int(args.get("imshow", '0'))
    ooi = args["ooi"]

    main(cwd, executable, script, path, enable_imshow, ooi)
