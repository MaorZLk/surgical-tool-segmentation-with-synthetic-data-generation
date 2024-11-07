import json
import os
import shutil
from tqdm import tqdm


PATH1 = '/home/student/Desktop/Visualization_project/synth_dataset'
PATH2 = '/home/student/Desktop/Visualization_project/synth_dataset_two'
for PATH in [PATH1, PATH2]:
    PATH_IMAGES = f'{PATH}/images'
    PATH_LABELS = f'{PATH}/labels'
    PATH_IMAGES_TRAIN = f'{PATH_IMAGES}/train'
    PATH_IMAGES_VAL = f'{PATH_IMAGES}/val'
    PATH_LABELS_TRAIN = f'{PATH_LABELS}/train'
    PATH_LABELS_VAL = f'{PATH_LABELS}/val'

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    if not os.path.exists(PATH_IMAGES):
        os.mkdir(PATH_IMAGES)

    if not os.path.exists(PATH_LABELS):
        os.mkdir(PATH_LABELS)

    if not os.path.exists(PATH_IMAGES_TRAIN):
        os.mkdir(PATH_IMAGES_TRAIN)

    if not os.path.exists(PATH_IMAGES_VAL):
        os.mkdir(PATH_IMAGES_VAL)

    if not os.path.exists(PATH_LABELS_TRAIN):
        os.mkdir(PATH_LABELS_TRAIN)

    if not os.path.exists(PATH_LABELS_VAL):
        os.mkdir(PATH_LABELS_VAL)


TRAIN_TEST_SPLIT = 0.7

def main():
    # setting paths
    PATH = '/home/student/Desktop/Visualization_project/synth_dataset'
    PATH_IMAGES = f'{PATH}/images'
    PATH_LABELS = f'{PATH}/labels'
    PATH_IMAGES_TRAIN = f'{PATH_IMAGES}/train'
    PATH_IMAGES_VAL = f'{PATH_IMAGES}/val'
    PATH_LABELS_TRAIN = f'{PATH_LABELS}/train'
    PATH_LABELS_VAL = f'{PATH_LABELS}/val'

    kinds = ['hdri', 'non_hdri']
    labels = {'needle_holder': 1,'tweezers': 2}
    for kind in kinds:
        for label in labels.keys():
            TEMP_PATH = f'/home/student/Desktop/Visualization_project/synthetic_data/{kind}/{label}/coco_data'
            with open(f'{TEMP_PATH}/coco_annotations.json') as f:
                df = json.load(f)
                # print(df.keys())
                # print(df['categories'][0])
                # print('\n\n')
                # print(df['images'][0])
                # print('\n\n')
                # print(df['annotations'][0]['height'])
                # print(len(df['annotations'][0]['segmentation'][0]))
                # print(len(df['images']))
                # print(len(df['annotations']))
                number_of_images = len(df['images'])
                
                for i in tqdm(range(int(len(df['images']) * TRAIN_TEST_SPLIT))):
                    file_name = f"{kind}_{label}_{df['images'][i]['file_name'].split('/')[-1].split('.')[0]}"
                    segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                                else str(seg / df['annotations'][i]['height'])
                                for k, seg in enumerate(df['annotations'][i]['segmentation'][0])]
                    segments.insert(0, str(labels[label]))
                    text = ' '.join(segments)
                    with open(f"{PATH_LABELS_TRAIN}/{file_name}.txt", "w") as f:
                        f.write(text)
                    
                    if kind == 'hdri':
                        shutil.copy(f"{TEMP_PATH}/{df['images'][i]['file_name']}", f"{PATH_IMAGES_TRAIN}/{file_name}.png")
                    else:
                        shutil.copy(f"{TEMP_PATH}/images/output/{df['images'][i]['file_name'].split('/')[-1]}", f"{PATH_IMAGES_TRAIN}/{file_name}.png")
                    
                    
                for i in tqdm(range(int(len(df['images']) * TRAIN_TEST_SPLIT), len(df['images']))):
                    file_name = f"{kind}_{label}_{df['images'][i]['file_name'].split('/')[-1].split('.')[0]}"
                    segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                                else str(seg / df['annotations'][i]['height'])
                                for k, seg in enumerate(df['annotations'][i]['segmentation'][0])]                
                    segments.insert(0, str(labels[label]))
                    text = ' '.join(segments)
                    with open(f"{PATH_LABELS_VAL}/{file_name}.txt", "w") as f:
                        f.write(text)
                    
                    if kind == 'hdri':
                        shutil.copy(f"{TEMP_PATH}/{df['images'][i]['file_name']}", f"{PATH_IMAGES_VAL}/{file_name}.png")
                    else:
                        shutil.copy(f"{TEMP_PATH}/images/output/{df['images'][i]['file_name'].split('/')[-1]}", f"{PATH_IMAGES_VAL}/{file_name}.png")
                
def multiple_tools():
    # setting paths
    PATH = '/home/student/Desktop/Visualization_project/synth_dataset_two'
    PATH_IMAGES = f'{PATH}/images'
    PATH_LABELS = f'{PATH}/labels'
    PATH_IMAGES_TRAIN = f'{PATH_IMAGES}/train'
    PATH_IMAGES_VAL = f'{PATH_IMAGES}/val'
    PATH_LABELS_TRAIN = f'{PATH_LABELS}/train'
    PATH_LABELS_VAL = f'{PATH_LABELS}/val'

    kinds = ['hdri', 'non_hdri']
    for kind in kinds:
        TEMP_PATH = f'/home/student/Desktop/Visualization_project/synthetic_data_two/{kind}/coco_data'
        with open(f'{TEMP_PATH}/coco_annotations.json') as f:
            df = json.load(f)
            # print(df.keys())
            # print(df['categories'][1])
            # print('\n\n')
            # print(df['images'][-1])
            # print('\n\n')
            # print(df['annotations'][0])
            # print(df['annotations'][1])
            # print(len(df['annotations'][0]['segmentation'][0]))
            # print(len(df['images']))
            # print(len(df['annotations']))
            # number_of_images = len(df['images'])
            for i in tqdm(range(int(len(df['images']) * TRAIN_TEST_SPLIT))):
                file_name = f"{kind}_{df['images'][i]['file_name'].split('/')[-1].split('.')[0]}"
                segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                            else str(seg / df['annotations'][i]['height'])
                            for k, seg in enumerate(df['annotations'][2 * i]['segmentation'][0])]
                segments.insert(0, str(df['annotations'][2 * i]['category_id']))
                text = ' '.join(segments)
                with open(f"{PATH_LABELS_TRAIN}/{file_name}.txt", "w") as f:
                    f.write(text)

                segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                            else str(seg / df['annotations'][i]['height'])
                            for k, seg in enumerate(df['annotations'][2 * i + 1]['segmentation'][0])]
                segments.insert(0, str(df['annotations'][2 * i + 1]['category_id']))
                text = ' '.join(segments)
                with open(f"{PATH_LABELS_TRAIN}/{file_name}.txt", "a") as f:
                    f.write(f'\n{text}')

                if kind == 'hdri':
                    shutil.copy(f"{TEMP_PATH}/{df['images'][i]['file_name']}", f"{PATH_IMAGES_TRAIN}/{file_name}.png")
                else:
                    shutil.copy(f"{TEMP_PATH}/images/output/{df['images'][i]['file_name'].split('/')[-1]}", f"{PATH_IMAGES_TRAIN}/{file_name}.png")
                
                
            for i in tqdm(range(int(len(df['images']) * TRAIN_TEST_SPLIT), len(df['images']))):
                file_name = f"{kind}_{df['images'][i]['file_name'].split('/')[-1].split('.')[0]}"
                segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                            else str(seg / df['annotations'][i]['height'])
                            for k, seg in enumerate(df['annotations'][2 * i]['segmentation'][0])]
                segments.insert(0, str(df['annotations'][2 * i]['category_id']))
                text = ' '.join(segments)
                with open(f"{PATH_LABELS_TRAIN}/{file_name}.txt", "w") as f:
                    f.write(text)

                segments = [str(seg / df['annotations'][i]['width'] ) if k % 2 == 0 
                            else str(seg / df['annotations'][i]['height'])
                            for k, seg in enumerate(df['annotations'][2 * i + 1]['segmentation'][0])]
                segments.insert(0, str(df['annotations'][2 * i + 1]['category_id']))
                text = ' '.join(segments)
                with open(f"{PATH_LABELS_TRAIN}/{file_name}.txt", "a") as f:
                    f.write(f'\n{text}')
                
                if kind == 'hdri':
                    shutil.copy(f"{TEMP_PATH}/{df['images'][i]['file_name']}", f"{PATH_IMAGES_VAL}/{file_name}.png")
                else:
                    shutil.copy(f"{TEMP_PATH}/images/output/{df['images'][i]['file_name'].split('/')[-1]}", f"{PATH_IMAGES_VAL}/{file_name}.png")
if __name__ == '__main__':
    main()
    multiple_tools()

