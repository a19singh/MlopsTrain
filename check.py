#!/usr/bin/python36

rfile = open('/root/Workspace/ML/mlcontainer/task3.py','r')
inpu = rfile.read()

if 'keras' in inpu:
    print('DL')
else:
    print('not DL')
