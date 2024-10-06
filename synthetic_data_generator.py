# import blenderproc as bproc
import os
import subprocess
from paste_on_random_background import main
# from render_on_hdri import main

outputs = []

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/hdri/needle_holder"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/hdri/needle_holder")

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/hdri/tweezers"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/hdri/tweezers")

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/needle_holder"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/needle_holder")

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/tweezers"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/tweezers")

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/with_background/needle_holder"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/with_background/needle_holder")

if not os.path.exists("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/with_background/tweezers"):
    os.makedirs("/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/with_background/tweezers")

os.environ['PATH'] = "/anaconda/envs/synth/bin:" + os.environ['PATH']

# os.system("conda activate synth")

# generating HDRI images
# Generate 500 synthetic images for needle_holder
obj_files = [f for f in os.listdir("/datashare/project/surgical_tools_models/needle_holder") if '.obj' in f]

for i in range(len(obj_files)):
    # obj_path = f"/datashare/project/surgical_tools_models/needle_holder/NH{i + 1}.obj"
    # num_images = f"{int(500 / len(obj_files))}"
    # subprocess.run(["/home/student/Desktop/Visualization_project/run_blenderproc.sh", obj_path, num_images])

    command = ' '.join(["blenderproc", "run", "/home/student/Desktop/Visualization_project/render_on_hdri.py",
                                                "--obj_name", "needle_holder",
                                                "--obj", f"/datashare/project/surgical_tools_models/needle_holder/NH{i + 1}.obj",
                                                "--camera_params", "/datashare/project/camera.json",
                                                "--output_dir", "/home/student/Desktop/Visualization_project/synthetic_data/hdri/needle_holder",
                                                "--num_images", f"{int(500 / len(obj_files))}",
                                                "--haven_path", "/datashare/project/haven/"])
    subprocess.run(command, shell=True, executable='/bin/bash')

# Generate 500 synthetic images for tweezers
obj_files = [f for f in os.listdir("/datashare/project/surgical_tools_models/tweezers") if '.obj' in f]

for i in range(len(obj_files)):
    command = ' '.join(["blenderproc", "run", "/home/student/Desktop/Visualization_project/render_on_hdri.py",
                                                 "--obj_name", "tweezers",
                                                "--obj", f"/datashare/project/surgical_tools_models/tweezers/T{i + 1}.obj",
                                                "--camera_params", "/datashare/project/camera.json",
                                                "--output_dir", "/home/student/Desktop/Visualization_project/synthetic_data/hdri/tweezers",
                                                "--num_images", f"{int(500 / len(obj_files))}",
                                                "--haven_path", "/datashare/project/haven/"])
    subprocess.run(command, shell=True, executable='/bin/bash')

#generating non-hdri images
# Generate 250 synthetic images for needle_holder
obj_files = [f for f in os.listdir("/datashare/project/surgical_tools_models/needle_holder") if '.obj' in f]

for i in range(len(obj_files)):
    command = ' '.join(["blenderproc", "run", "/home/student/Desktop/Visualization_project/render_before_backgorund.py",
                                                "--obj_name", "needle_holder",
                                                "--obj", f"/datashare/project/surgical_tools_models/needle_holder/NH{i + 1}.obj",
                                                "--camera_params", "/datashare/project/camera.json",
                                                "--output_dir", "/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/needle_holder",
                                                "--num_images", f"{int(250 / len(obj_files))}"])
    subprocess.run(command, shell=True, executable='/bin/bash')
    

# Generate 250 synthetic images for tweezers
obj_files = [f for f in os.listdir("/datashare/project/surgical_tools_models/tweezers") if '.obj' in f]
for i in range(len(obj_files)):
    command = ' '.join(["blenderproc", "run", "/home/student/Desktop/Visualization_project/render_before_backgorund.py",
                                                "--obj_name", "tweezers",
                                                "--obj", f"/datashare/project/surgical_tools_models/tweezers/T{i + 1}.obj",
                                                "--camera_params", "/datashare/project/camera.json",
                                                "--output_dir", "/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/tweezers",
                                                "--num_images", f"{int(250 / len(obj_files))}"])
    subprocess.run(command, shell=True, executable='/bin/bash')
    

command = ' '.join(["python", "/home/student/Desktop/Visualization_project/paste_on_random_background.py", 
                    '-i', '/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/needle_holder/coco_data/images',
                    '-b', '/datashare/project/train2017'])
subprocess.run(command, shell=True, executable='/bin/bash')

command = ' '.join(["python", "/home/student/Desktop/Visualization_project/paste_on_random_background.py", 
                    '-i', '/home/student/Desktop/Visualization_project/synthetic_data/non_hdri/tweezers/coco_data/images',
                    '-b', '/datashare/project/train2017',])
subprocess.run(command, shell=True, executable='/bin/bash')

# main(['-i', '/home/student/Desktop/Visualization_project/synthetic_data/needle_holder',
#      '-b', '/datashare/project/train2017',
#      '--output', '/home/student/Desktop/Visualization_project/synthetic_data/with_background/needle_holder'])

# main(['-i', '/home/student/Desktop/Visualization_project/synthetic_data/tweezers',
#      '-b', '/datashare/project/train2017',
#      '-o', '/home/student/Desktop/Visualization_project/synthetic_data/with_background/tweezers'])

# python examples/render_and_paste/paste_on_random_background.py -i coco_data/images -b /datashare/project/train2017
