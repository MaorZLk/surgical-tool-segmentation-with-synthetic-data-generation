from ultralytics import YOLO
from PIL import Image
import shutil
import os

SYNTH_TRAINED_WEIGHTS = '/home/student/Desktop/Visualization_project/runs/segment/train/weights/best.pt' # TODO add
PATH_HW1_DATA = '/datashare/HW1/labeled_image_data'
PATH_CROPPED_HW1_IMAGES = '/home/student/Desktop/Visualization_project/cropped_images'
PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL = '/home/student/Desktop/Visualization_project/cropped_images_psudo_labeled'
PATH_DATASET_TRAIN_IMAGES = '/home/student/Desktop/Visualization_project/synth_dataset/images/train'
PATH_DATASET_TRAIN_LABELS = '/home/student/Desktop/Visualization_project/synth_dataset/labels/train'
PATH_ID_VIDEOS = '/datashare/HW1/id_video_data'
PATH_OOD_VIDEOS = '/datashare/HW1/ood_video_data'
PATH_VIDEO_FRAMES = '/home/student/Desktop/Visualization_project/video_frames'

if not os.path.exists(PATH_CROPPED_HW1_IMAGES):
    os.mkdir(PATH_CROPPED_HW1_IMAGES)

if not os.path.exists(PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL):
    os.mkdir(PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL)

model = YOLO(SYNTH_TRAINED_WEIGHTS)

'''
************************************************
1. First cut the HW1 pictures using bounding box
************************************************
'''

def crop_image_to_bbox(image_path, bbox):
    """
    Crop an image to the bounding box region.
    
    Args:
        image_path (str): Path to the input image file.
        bbox (list): Bounding box coordinates in the format [x_center, y_center, width, height].
                     Coordinates are normalized between 0 and 1.

    Returns:
        cropped_img (PIL.Image): Cropped image based on the bounding box.
    """
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Extract bounding box information (assuming normalized [0, 1] format)
    x_center, y_center, width, height = bbox
    
    # Convert normalized coordinates to pixel values
    x_center = int(x_center * img_width)
    y_center = int(y_center * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    
    # Calculate the top-left corner of the bounding box
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    
    # Calculate the bottom-right corner
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    # Ensure the bounding box coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    
    # Crop the image using the bounding box
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    
    return cropped_img

categories = []
HW1_images = []
HW1_categories = []
for ver in ['train','val']:
    for image in os.listdir(f"{PATH_HW1_DATA}/images/{ver}"):
        image_name = image.split(".")[0]
        with open(f"{PATH_HW1_DATA}/labels/{ver}/{image_name}.txt", 'r') as f:
            text = f.read()
            bboxes = text.split('\n')
            if '' in bboxes:
                bboxes = bboxes[:-1]
            for bbox in bboxes:
                if int(bbox.split(' ')[0]) == 0:
                    continue
                bbox_list = [float(b) for b in bbox.split(' ')[1:]]
                categories.append(bbox.split(' ')[0])
                cropped_image = crop_image_to_bbox(f"{PATH_HW1_DATA}/images/{ver}/{image}", bbox_list)
                cropped_image.save(f"{PATH_CROPPED_HW1_IMAGES}/{image}")
                HW1_images.append(f"{PATH_CROPPED_HW1_IMAGES}/{image}")
                break
        HW1_categories.append(f"{PATH_HW1_DATA}/labels/{ver}/{image_name}.txt")


'''
************************************************
5. Add training loops only on psudo labels
************************************************
'''

def is_segmentation_inside_bbox(segmentation, bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # Iterate over each point in the segmentation polygon
    for i in range(0, len(segmentation), 2):
        x, y = segmentation[i], segmentation[i + 1]

        # Check if the point is within the bounding box
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False  # If any point is outside, return False

    return True  # All points are inside

def clear_synth_dataset():
    # Iterate over all files in the directory
    for filename in os.listdir(PATH_DATASET_TRAIN_IMAGES):
        file_path = os.path.join(PATH_DATASET_TRAIN_IMAGES, filename)
        
        # Check if it's a file (and not a directory), then delete it
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Iterate over all files in the directory
    for filename in os.listdir(PATH_DATASET_TRAIN_LABELS):
        file_path = os.path.join(PATH_DATASET_TRAIN_LABELS, filename)
        
        # Check if it's a file (and not a directory), then delete it
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_synth_dataset()
from prepering_synthetic_data import main as prep_synth

prep_synth()
for i in range(0,20):
    SYNTH_TRAINED_WEIGHTS = f"/home/student/Desktop/Visualization_project/runs/segment/train{i + 1 if i > 0 else ''}/weights/best.pt"
    model = YOLO(SYNTH_TRAINED_WEIGHTS)

    if i < 10:
        num_freeze = len(list(model.model.modules()))*0.4
    elif i > 10:
        num_freeze = len(list(model.model.modules()))*0.3

    '''
    ************************************************
    2. Generate Pseudo-Labels
    ************************************************
    '''

    # * predict images
    images = [f"{PATH_CROPPED_HW1_IMAGES}/{f}" for f in os.listdir(PATH_CROPPED_HW1_IMAGES)]

    results = model(images, task='segment', 
                    iou=0.7 if i < 20 else 0.3,
                    conf=0.25 if i < 20 else 0.5,
                    device=0)

    for i, result in enumerate(results):
        result.save_txt(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt")

        wrong_input = False

        if os.path.isfile(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt"):
            classification = []
            with open(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt", "r") as f:
                text = f.read().split('\n')
                for c in text:
                    temp = c.split(' ')
                    if len(c) > 1:
                        classification.append(c)
            text = '\n'.join(classification)
            with open(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt", "w") as f:
                f.write(text)
        
        # * comparing input based on given bounding box
        fixed_seg_text = []
        if os.path.isfile(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt"):
            with open(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt") as seg_file:
                segmentation_txt = seg_file.read().split('\n')
                with open(HW1_categories[i], 'r') as bb_file:
                    bb_txt = bb_file.read().split('\n')

                    for seg_obj in segmentation_txt:
                        for bb_obj in bb_txt:
                            if (seg_obj.split(' ')[0] != '' and bb_obj.split(' ')[0] != 0) and \
                            seg_obj.split(' ')[0] == bb_obj.split(' ')[0] and \
                            is_segmentation_inside_bbox(seg_obj.split(' ')[1:], bb_obj.split(' ')[1:]):
                                fixed_seg_text.append(seg_obj)

            text = '\n'.join(fixed_seg_text)
            with open(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt",'w') as seg_file:
                seg_file.write(text)
                
        if len(fixed_seg_text) == []:
            os.remove(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.txt")
            continue
        
        result.save(filename=f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/labeled_hw1_image_{i}.png")
        Image.open(images[i]).save(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/hw1_image_{i}.png")
    

    # * predict video frames
    # for video in os.listdir(PATH_VIDEO_FRAMES):
    #     images_full = [f"{PATH_VIDEO_FRAMES}/{video}/{f}" for f in os.listdir(f'{PATH_VIDEO_FRAMES}/{video}')]
    #     for images in [images_full[:int(len(images_full)/2)], images_full[int(len(images_full)/2):]]:
    #         results = model(images, task='segment', 
    #                 iou=0.7 if i < 20 else 0.3,
    #                 conf=0.25 if i < 20 else 0.5,
    #                 device=0)
            
    #         for j, result in enumerate(results):
    #             i = j + 1
    #             if j >= 50:
    #                 break
    #             # Get the predicted class from YOLOv8 results
    #             # predicted_class = result.names[result.probs[0, :, 5].argmax()]  # The class with max confidence
    #             result.save_txt(f"{PATH_DATASET_TRAIN_IMAGES}/{video}-frame_{i}.txt")

    #             wrong_input = False

    #             if os.path.isfile(f"{PATH_DATASET_TRAIN_IMAGES}/{video}-frame_{i}.txt"):
    #                 classification = []
    #                 with open(f"{PATH_DATASET_TRAIN_IMAGES}/{video}-frame_{i}.txt", "r") as f:
    #                     text = f.read().split('\n')
    #                     for c in text:
    #                         temp = c.split(' ')
    #                         if len(c) > 1:
    #                             classification.append(c)
    #                 text = '\n'.join(classification)
    #                 with open(f"{PATH_DATASET_TRAIN_IMAGES}/{video}-frame_{i}.txt", "w") as f:
    #                     f.write(text)

    #                 shutil.copy(f"{PATH_VIDEO_FRAMES}/{video}/{video}-frame_{i}.jpg", f"{PATH_DATASET_TRAIN_IMAGES}/{video}-frame_{i}.jpg")
    

    '''
    ************************************************
    3. Prepare images for new 
    ************************************************
    '''
    psuedo_images = []
    for image in os.listdir(PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL):
        if '.txt' not in image:
            continue

        psuedo_images.append(image)
        shutil.move(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/{image}", f"{PATH_DATASET_TRAIN_LABELS}/{image}")
        shutil.move(f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/{image.split('.')[0]}.png",f"{PATH_DATASET_TRAIN_IMAGES}/{image.split('.')[0]}.png")

    '''
    ************************************************
    4. Train on new dataset and return psudo images to original location
    ************************************************
    '''
    try:
        if i % 5 != 0:
            results = model.train(data="/home/student/Desktop/Visualization_project/training_paths.yml",
                                epochs=50, device=0, freeze=num_freeze)
        else:
            results = model.train(data="/home/student/Desktop/Visualization_project/training_paths2.yml", 
                                epochs=50, device=0, freeze=len(list(model.model.modules()))*0.7)
    except FileNotFoundError:
        results = model.train(data="/home/student/Desktop/Visualization_project/training_paths2.yml",
                            epochs=50, device=0, freeze=num_freeze)


    for image in psuedo_images:
        shutil.move(f"{PATH_DATASET_TRAIN_LABELS}/{image}", f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/{image}")
        shutil.move(f"{PATH_DATASET_TRAIN_IMAGES}/{image.split('.')[0]}.png", f"{PATH_CROPPED_HW1_IMAGES_PSUDO_LABEL}/{image.split('.')[0]}.png")

    clear_synth_dataset()
    if i % 6 == 0:
        prep_synth()