# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Sequence

from monai import data
from monai.data import load_decathlon_datalist

from transforms import compose_train_transform, compose_test_transform

__all__ = ["get_train_loader",
           "get_test_loader"]


def get_train_loader(data_dir: str,
                     json_list: str,
                     batch_size: int,
                     space: Sequence[float] | float,
                     roi_size: Sequence[int] | int,
                     a_min: float,
                     a_max: float,
                     b_min: float,
                     b_max: float,
                     ) -> list[data.DataLoader]:
    train_transform = compose_train_transform(
        space=space,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        roi_size=roi_size,
    )
    test_transform = compose_test_transform(
        space=space,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
    )

    data_dir = data_dir
    datalist_json = os.path.join(data_dir, json_list)

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    train_ds = data.Dataset(data=datalist, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds, batch_size=1, shuffle=False, pin_memory=True
    )
    return [train_loader, test_loader]


def get_test_loader(data_dir: str,
                    json_list: str,
                    space: Sequence[float] | float,
                    a_min: float,
                    a_max: float,
                    b_min: float,
                    b_max: float):
    data_dir = data_dir
    datalist_json = os.path.join(data_dir, json_list)

    test_transform = compose_test_transform(
        space=space,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
    )

    test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return test_loader
