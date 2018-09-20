import shutil, os
from glob import glob

RAW_IMG_PATH = '/home/public/split/foshan/batch/1/0157/'
HAS_PATH = 'has/'
HASNOT_PATH = 'hasnot/'

pl = glob(HASNOT_PATH + '*.jp*')

spec_count = 0
for p in pl:
    bn = os.path.basename(p)
    path = RAW_IMG_PATH + bn
    if os.path.exists(path):
        shutil.copy(path, HAS_PATH)
        print(bn)
