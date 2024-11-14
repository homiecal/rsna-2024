from collections import namedtuple
import functools
import datetime
import glob
from torch.utils.data import Dataset
import random
import pydicom
import re


candidate_info_tuple = namedtuple(
    "candidate_info_tuple",
    "row_id, study_id, series_id, instance_number, centre_xy, severity, img_path, width_bbox, height_bbox",
)
SEVERITY_MAPPING = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
SEVERITY_WEIGHTING = {0: 1.0, 1: 2.0, 2: 4.0}


@functools.lru_cache(1, typed=True)
def get_image(path):
    return pydicom.dcmread(path)


def clean_newlines(input_string):
    # Replace multiple newlines with a single newline
    cleaned_string = re.sub(r'\n+', '\n', input_string)
    # If there are newlines at the end, remove them
    cleaned_string = re.sub(r'\n+$', '', cleaned_string)
    return cleaned_string.split("\n")

# Function to extract the numeric values from the filename
def extract_numbers(file_path):
    filename = file_path.split('/')[-1]
    numbers = re.findall(r'\d+', filename)
    return list(map(int, numbers))

