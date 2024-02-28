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

from monai import data
from monai.data import load_decathlon_datalist

from transforms import compose_train_transform, compose_test_transform


def get_train_loader(args) -> list[data.DataLoader]:
    train_transform = compose_train_transform(
        space=(args.space_x, args.space_y, args.space_z),
        a_min=args.a_amin,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
    )
    test_transform = compose_test_transform(
        space=(args.space_x, args.space_y, args.space_z),
        a_min=args.a_amin,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
    )

    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    if args.use_normal_dataset:
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    return [train_loader, test_loader]


def get_test_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    test_transform = compose_test_transform(
        space=(args.space_x, args.space_y, args.space_z),
        a_min=args.a_amin,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
    )

    test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return test_loader
