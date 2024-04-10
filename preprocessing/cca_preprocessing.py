import argparse
import os

from skimage import measure
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="")


def get_largest_component(component_map, component_num):
    max_vol, max_i = 0, 0
    
    for i in range(1, component_num + 1):
        vol = np.count_nonzero(component_map == i)
        if vol > max_vol:
            max_vol = vol
            max_i = i
            
    return max_i


def main(args):
    from_dir = os.path.join(args.data_dir, "labels")
    to_dir = os.path.join(args.data_dir, "primary_lesion_labels")

    files = os.listdir(from_dir)

    os.makedirs(to_dir, exist_ok=True)

    for file in tqdm(files):
        from_path = os.path.join(from_dir, file)
        to_path = os.path.join(to_dir, file)

        label = np.load(from_path)
        assert label.dtype == int

        omtm_component_map, omtm_component_num = measure.label((label == 1).astype(int), return_num=True)
        omtm_max = get_largest_component(omtm_component_map, omtm_component_num)

        pod_component_map, pod_component_num = measure.label((label == 2).astype(int), return_num=True)
        pod_max = get_largest_component(pod_component_map, pod_component_num)

        new_label = np.zeros_like(label)
        if omtm_max:
            new_label[omtm_component_map == omtm_max] = 1
        if pod_max:
            new_label[pod_component_map == pod_max] = 2

        np.save(to_path, new_label)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
