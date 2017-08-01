import os
import glob
import shutil

dir_name = 'gif3'
images = glob.glob('{}/*.jpg'.format(dir_name))
print(images)

new_dir = '{}_sub'.format(dir_name)
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)

os.makedirs(new_dir)

subsample_ratio = 22

for i in range(0, len(images), subsample_ratio):
    shutil.copy(images[i], os.path.join(new_dir, os.path.basename(images[i])))
