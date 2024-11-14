import argparse
from helpers.constants import DATA_PATH, WORKING_DIR
from helpers.img_data import get_image, candidate_info_tuple, extract_numbers
from pandas import pd
from typing import Dict,Union
import os
import glob

from collections import namedtuple
import functools
import datetime
from torch.utils.data import Dataset
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from torch import tensor
import pickle

!pip install ultralytics # '8.3.23'
import os 
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO, settings
settings.update({"wandb": False})


LUMBAR_LABEL_MAPPING = {"l1_l2": 0, "l2_l3": 1, "l3_l4": 2, "l4_l5": 3, "l5_s1": 4}
LUMBAR_INT_MAPPING = {value: key for key, value in LUMBAR_LABEL_MAPPING.items()}

AXIAL_LABEL_MAPPING = {"left": 0, "right": 1}
AXIAL_INT_MAPPING = {value: key for key, value in LUMBAR_LABEL_MAPPING.items()}

SEVERITY_MAPPING = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
SEVERITY_WEIGHTING = {0: 1.0, 1: 2.0, 2: 4.0}

IMG_RESIZE = (416, 416)


import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import pydicom
import numpy as np
from PIL import Image
from random import randint
from torch.utils.data import DataLoader

import torch


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--target", type=str, default="axial", choices=["axial", "sagittal"]
    )
    args = parser.parse_args()
    return args


def get_data(set_type: str) -> Dict[pd.DataFrame]:
    set_desc = pd.read_csv(DATA_PATH + set_type + "_series_descriptions.csv")
    sagittal_t1 = set_desc[set_desc["series_description"] == "Sagittal T1"]
    sagittal = set_desc[
        set_desc["series_description"].isin(["Sagittal T2/STIR", "Sagittal T1"])
    ]
    axial = set_desc[set_desc["series_description"].isin(["Axial T2"])]
    datasets = {
        "sagittal": sagittal,
        "axial": axial,
        "sagittal_t1": sagittal_t1,
        "all": set_desc,
    }

    return datasets


def get_study_id_meta(set_desc, set_type: str = "train"):
    """
    {
        study_id: {
            folder_path: ...
            series_ids: []
            series_descriptions: []
        }
    }
    """

    if set_type not in ("train", "test"):
        raise Exception(
            f"set_type must be equal to 'train' or 'test', set_type = '{set_type}''"
        )
    # First aggregate the descriptions, so we have a list of series descriptions, sort the dataframe by series_id to maintain order mapping
    set_desc_sorted = (
        set_desc.sort_values(["study_id", "series_id"]).groupby("study_id").agg(list)
    )
    test_study_ids = set_desc_sorted.index.tolist()

    # All file ids
    test_study_ids_path = [
        int(i.split("/")[-1])
        for i in glob.glob(os.path.join(DATA_PATH, f"{set_type}_images/*"))
    ]

    # Check all study_ids exist if they don't then delete from inference pack
    for ndx, id in enumerate(test_study_ids):
        if id not in test_study_ids_path:
            print(f"study_id {id} not on file")
            del test_study_ids[ndx]

    # Now create meta information, filtering on index
    study_id_meta = {}
    for study_id in test_study_ids:
        folder_path = os.path.join(DATA_PATH, f"{set_type}_images/{study_id}")
        study_id_meta.update(
            {
                study_id: {
                    "folder_path": folder_path,
                    "series_id_files": set_desc_sorted.loc[study_id].series_id,
                    "series_descriptions": set_desc_sorted.loc[
                        study_id
                    ].series_description,
                }
            }
        )

    return study_id_meta


@functools.lru_cache(1, typed=True)
def get_candidate_list(set_type: str, data_path: str, selection: str = "all"):

    candidate_dataset = datasets[selection]

    candidate_list = []

    for ndx, instance in enumerate(candidate_dataset.itertuples(name="Candidate")):

        try:
            img_path = f"{data_path}/{set_type}_images/{instance.study_id}/{instance.series_id}/"

            images = glob.glob(f"{img_path}/*.dcm")

            for inst_path in images:

                instance_number = inst_path.split("/")[-1].replace(".dcm", "")

                candidate_list.append(
                    candidate_info_tuple(
                        row_id=None,
                        study_id=instance.study_id,
                        series_id=instance.series_id,
                        instance_number=instance_number,
                        centre_xy=tuple(),
                        severity=None,
                        img_path=inst_path,
                        width_bbox=None,
                        height_bbox=None,
                    )
                )

            if not os.path.exists(img_path):
                print(f"Path not found {img_path}")
                continue

            if ndx % 10000 == 0 and ndx != 0:
                print(f"{datetime.datetime.now()}, {ndx} records processed")

        except Exception as e:
            print(instance)
            raise e

    return candidate_list


def yolo_preprocess(img: np.array, img_resize: tuple):
    """ """

    # Normalize to the range [0, 255] using the actual max value
    image_2d = img.astype(float)
    image_2d = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    image_2d = np.uint8(image_2d)

    # Resize image
    resized_image = cv2.resize(image_2d, IMG_RESIZE, interpolation=cv2.INTER_LINEAR)

    return resized_image


class LumbarYoloInference(Dataset):

    def __init__(
        self,
        val_stride=0,
        val_set=False,
        study_id=None,
        candidate_list=None,
        transform=None,
        sample: int | None = None,
        rand_bbox: bool = False,
        sorted_candidate_list=None,
        save_to_drive: bool = False,
    ):

        self.sample = sample
        self.rand_bbox = rand_bbox
        self.save_to_drive = save_to_drive

        if candidate_list:
            self.candidate_list = candidate_list.copy()
        else:
            self.candidate_list = get_candidate_list().copy()

        # Default to a stratified sampled of candidates by study_id (option to provide custom sorted list on other
        # stratifications).
        if sorted_candidate_list:
            self.sorted_candidate_list = sorted_candidate_list.copy()
        else:
            study_ids = np.unique([x.study_id for x in self.candidate_list])
            self.sorted_candidate_list = np.random.shuffle(study_ids)

        if study_id:
            self.candidate_list = [
                x for x in self.candidate_list if x.study_id == study_id
            ]

        if val_set:
            assert val_stride > 0, val_stride
            val_study_ids = self.sorted_candidate_list[::val_stride]
            self.candidate_list = [
                x for x in self.candidate_list if str(x.study_id) in val_study_ids
            ]
            assert self.candidate_list
        elif val_stride > 0:
            del self.sorted_candidate_list[::val_stride]
            self.candidate_list = [
                x
                for x in self.candidate_list
                if str(x.study_id) in self.sorted_candidate_list
            ]
            assert self.candidate_list

        self.transform = transform

    def __len__(self):
        if self.sample:
            return min(self.sample, len(self.candidate_list))
        else:
            return len(self.candidate_list)

    def __getitem__(self, ndx):
        candidate = self.candidate_list[ndx]

        # Get full image
        image = get_image(candidate.img_path)

        resized_image = yolo_preprocess(image.pixel_array, IMG_RESIZE)

        # Pass in defaults so can be used with DataLoader
        candidate = candidate._replace(
            row_id=candidate.row_id or "",
            severity=candidate.severity or 0,
            centre_xy=candidate.centre_xy or (0, 0),
            width_bbox=candidate.width_bbox or 0,
            height_bbox=candidate.height_bbox or 0,
        )
        return (
            resized_image,
            str(candidate.study_id),
            str(candidate.series_id),
            candidate.instance_number,
        )


def save_image(image, study_id, series_id, instance_number, inference_dir):
    image_key = "_".join([str(study_id), str(series_id), str(int(instance_number))])
    img_path = f"{inference_dir}{image_key}.png"

    # Save image
    cv2.imwrite(img_path, image.numpy())


def create_yolo_dataset(inference_set: LumbarYoloInference, **kwargs):

    InferenceLoader = DataLoader(dataset=inference_set, batch_size=64, num_workers=4)

    for ndx, batch in enumerate(InferenceLoader):

        images, study_id, series_id, instance_number = batch

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(
                save_image, images, study_id, series_id, instance_number, **kwargs
            )

        if (ndx * 64) % 1280 == 0:
            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Processed {ndx*64} images."
            )

        if ndx == len(InferenceLoader):
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, End")


def run_predictions(files_for_inference: list, study_id_meta: dict, model, inf_batch_size: int = 100):

    boxes_meta = {}
    class_list = [0,1,2,3,4]
    for key, values in study_id_meta.items():
        for series in values['series_id_files']:
            boxes_meta.update({f"{key}_{series}": {i:{'instance_no':[],"box_coord":[],"box_conf":[], "left_right":[]} for i in class_list}})


    no_batches = (len(files_for_inference)//inf_batch_size)+1
    batches_files_for_inference = [files_for_inference[((i*inf_batch_size)):(i+1)*inf_batch_size if (i+1) < no_batches else len(files_for_inference)] for i in range(no_batches)]
    assert batches_files_for_inference[no_batches-1][-1] == files_for_inference[-1]

    torch.cuda.empty_cache()

    for ndx, files in enumerate(batches_files_for_inference):
        
        results = model(files,task='detect',iou=0.8,stream=True, cache=False, batch=1, save=False, verbose=False)  # return a list of Results objects
        
        for slice_result in results:
            if slice_result.boxes:
                img_path = slice_result.path
                study_id, series_id, instance_no = extract_numbers(img_path)
                box = slice_result.boxes
                classes = box.cls.cpu()
                conf = box.conf.cpu()
                coord = box.xywh.cpu()
                id_key = f"{study_id}_{series_id}"
                for i, class_ in enumerate(classes):
                    class_ = int(class_)
                    boxes_meta[id_key][class_]['instance_no'].append(instance_no)
                    boxes_meta[id_key][class_]['box_conf'].append(conf[i])
                    boxes_meta[id_key][class_]['box_coord'].append(coord[i])
        
        if ndx%10==0:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Processed {ndx*inf_batch_size} images.")

        return boxes_meta
    


def get_series_ids_by_dataset(study_id_meta: dict, dataset: str='Sagittal T1') -> list:
    series_ids = [
        f"{key}_{series_id}"
        for key, entry in study_id_meta.items()
        for series_id, description in zip(entry['series_id_files'], entry['series_descriptions'])
        if description == dataset
    ]
    return series_ids


def rescale_bbox(bbox_xyxy: np.array, original_w_h=()):    
    # Original image size
    original_width = original_w_h[0]
    original_height = original_w_h[1]

    # Resized image size
    resized_width = IMG_RESIZE[0]
    resized_height = IMG_RESIZE[1]

    # Scaling factors
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    # Upscale the bounding box coordinates to the original image size
    bbox_original = bbox_xyxy * np.array([scale_x, scale_y, scale_x, scale_y])

    # bbox_original now contains the upscaled coordinates
    x1_upscaled, y1_upscaled, x2_upscaled, y2_upscaled = bbox_original
    
    return (int(x1_upscaled), int(y1_upscaled), int(x2_upscaled), int(y2_upscaled))


def get_xyxy(boxes: Union[list,torch.Tensor], return_int=True) -> tuple:
    """
    Get largest region in xyxy coordinates given a set of multiple slices of xywh coordinates.
    """
    if isinstance(boxes,list):
        boxes = torch.stack(boxes)
    elif isinstance(boxes, torch.Tensor) and len(boxes.shape)==1:
        boxes = boxes.unsqueeze(0)

    x_centers = boxes[:, 0]
    y_centers = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]

    x_mins = x_centers - widths / 2
    y_mins = y_centers - heights / 2
    x_maxs = x_centers + widths / 2
    y_maxs = y_centers + heights / 2

    # Compute the desired max and min values
    min_x = torch.min(x_mins).item()
    max_x = torch.max(x_maxs).item()
    min_y = torch.min(y_mins).item()
    max_y = torch.max(y_maxs).item()

    if return_int:
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)
    
    return (min_x, max_x, min_y, max_y)


def get_box_stats(data: dict):
    if data['instance_no']:
        data['roi_xyxy'] = get_xyxy(data['box_coord'])
        max_conf = torch.argmax(torch.tensor(data['box_conf'])).item() 
        data['conf_xyxy'] = get_xyxy(data['box_coord'][max_conf])
        data['conf_instance_no'] = data['instance_no'][max_conf]
    else:
        data['roi_xyxy'] = ()
        data['conf_xyxy'] = ()
        data['conf_instance_no'] = np.NaN
        
def arrange_sagittal_boxes_meta(boxes_meta: dict, study_id_meta: dict, set_type: str,  sagittal_t1_series_ids: list):

    # Approach (2) here
    new_data = {}
    for study_series, items in boxes_meta.items():
        preds = items
    #     for preds in items.values(): # Not at series level yet

        # Separate classes into left/right based on position of instance number
        for level in list(LUMBAR_LABEL_MAPPING.values()):
            
            level_class = LUMBAR_INT_MAPPING[level]
                    
            if study_series not in sagittal_t1_series_ids:
                inner_value = preds[level]
                get_box_stats(inner_value)
                
                new_data[f'{study_series}_spinal_canal_stenosis_{level_class}'] = inner_value      
                
            else:
            
                if level in preds.keys() and preds[level]['instance_no']:
                    instance_nos = preds[level]['instance_no']
                    try:
                        avg_in = np.mean(instance_nos)
                    except Exception:
                        print(instance_nos)
                    preds[level]['left_right'] = ["left" if i > avg_in else "right" for i in instance_nos]

                inner_value = preds[level]
                left_data = {'instance_no': [], 'box_coord': [], 'box_conf': [], 'roi_xyxy': ()}
                right_data = {'instance_no': [], 'box_coord': [], 'box_conf': [], 'roi_xyxy': ()}

                for i, lr in enumerate(inner_value['left_right']):
                    if lr == 'left':
                        left_data['instance_no'].append(inner_value['instance_no'][i])
                        left_data['box_coord'].append(inner_value['box_coord'][i])
                        left_data['box_conf'].append(inner_value['box_conf'][i])
                    elif lr == 'right':
                        right_data['instance_no'].append(inner_value['instance_no'][i])
                        right_data['box_coord'].append(inner_value['box_coord'][i])
                        right_data['box_conf'].append(inner_value['box_conf'][i])

                if right_data['box_coord']:
                    get_box_stats(right_data)

                if left_data['box_coord']:
                    get_box_stats(left_data)

                if left_data['instance_no']:
                    new_data[f'{study_series}_left_neural_foraminal_narrowing_{level_class}'] = left_data
                if right_data['instance_no']:
                    new_data[f'{study_series}_right_neural_foraminal_narrowing_{level_class}'] = right_data


    for key, value in new_data.items():
        study_id, series_id = tuple(key.split("_")[0:2])
        instance = value['conf_instance_no']
        if np.isnan(instance):
            continue
        img_path = f"{DATA_PATH}/{set_type}_images/{study_id}/{series_id}/{instance}.dcm"
        temp = get_image(img_path)
        
        set_type = study_id_meta[int(study_id)]['series_descriptions'][
                        study_id_meta[int(study_id)]['series_id_files'].index(int(series_id))
        ]
        
        # TODO update image path and all that other good stuff here
        value['img_path'] = img_path
        value['study_id'] = study_id
        value['series_id'] = series_id
        value['set_type'] = set_type

        value['roi_xyxy'] = rescale_bbox(np.array(value['roi_xyxy']), (temp.Rows, temp.Columns))
        value['conf_xyxy'] = rescale_bbox(np.array(value['conf_xyxy']), (temp.Rows, temp.Columns))
    




if __name__ == "__main__":

    args = get_args()

    set_type = args.mode
    dataset_type = args.target

    datasets = get_data(set_type)

    study_id_meta = get_study_id_meta(datasets[dataset_type], set_type)

    candidate_list = get_candidate_list(set_type, DATA_PATH, dataset_type)

    InferenceSet = LumbarYoloInference(candidate_list=candidate_list)

    yolo_dirs = ["/kaggle/working/yolo_set/images/inference/"]
    for d in yolo_dirs:
        if os.path.exists(d):
            continue
        else:
            os.makedirs(d)

    create_yolo_dataset(inference_set=InferenceSet, inference_dir=yolo_dirs[0])

    if set_type == 'sagittal':
        MODEL_DIR = "/kaggle/input/rsna-yolo-v1/pytorch/best_20240716/2/best_sagittal_20241105.pt"
    elif set_type == 'axial':
        MODEL_DIR = "/kaggle/input/rsna-yolo-v1/pytorch/best_20240716/2/best_sagittal_20241105.pt"

    files_for_inference = sorted(glob.glob(f"{yolo_dirs[0]}*"))

    # Load a model
    model = YOLO(MODEL_DIR)  # pretrained YOLOv8n model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # get files for inference, and pass model
    boxes_meta = run_predictions(files_for_inference=[], 
                    study_id_meta=study_id_meta,
                    model=model)

    sagittal_ids = get_series_ids_by_dataset(study_id_meta,"Sagittal T1")

    new_data = arrange_sagittal_boxes_meta(boxes_meta, study_id_meta, set_type, sagittal_ids)

    with open(f'/kaggle/working/{set_type}_sagittal_boxes.pkl', 'wb') as file:
        pickle.dump(new_data, file)
