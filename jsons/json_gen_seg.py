import json
import os
import csv
import random
import argparse
import glob


# Parse the input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="JSON file creator from csv file")
    parser.add_argument("--csv-file", default=None, type=str, help="csv file containing patient data")
    parser.add_argument("--json-file", default="data.json", type=str, help="json file to store patient data")
    parser.add_argument("--data-dir", default="data", type=str, help="json file to store patient data")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the data or not")
    parser.add_argument("--num-train", type=int, help="number of training datapoints", required=True)
    parser.add_argument("--num-val", type=int, help="number of validation datapoints", required=True)
    parser.add_argument("--num-test", type=int, help="number of testing datapoints")
    parser.add_argument("--data-name", default="dataset", help="name of the dataset")
    parser.add_argument("--pre-dir", default="pre_treatment", type=str)
    parser.add_argument("--post-dir", default="post_treatment", type=str)
    parser.add_argument("--image-dir", default="images", type=str)
    parser.add_argument("--label-dir", default="labels", type=str)
    parser.add_argument("--image-key", default="image", type=str)
    parser.add_argument("--label-key", default="label", type=str)
    parser.add_argument("--file-extension", default=".nii.gz", type=str)

    args = parser.parse_args()
    return args


def from_csv(args):
    csv_pth = os.path.join(args["data_dir"], args["csv_file"])

    return_paths = []
    with open(csv_pth, "r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

        for row in reader:
            info = dict(zip(header, row))

            pre_file = info["pre_treatment"]
            post_file = info["post_treatment"]

            return_paths += [
                {
                    args["image_key"]: os.path.join(args["pre_dir"], args["image_dir"], pre_file),
                    args["label_key"]: os.path.join(args["pre_dir"], args["label_dir"], pre_file)
                },
                {
                    args["image_key"]: os.path.join(args["post_dir"], args["image_dir"], post_file),
                    args["label_key"]: os.path.join(args["post_dir"], args["label_dir"], post_file)
                }
            ]

    return return_paths


def from_dir(args):
    pre_images = glob.glob(os.path.join(args["pre_dir"], args["image_dir"], f"*{args['file_extension']}"),
                           root_dir=args["data_dir"], recursive=True)
    pre_labels = glob.glob(os.path.join(args["pre_dir"], args["label_dir"], f"*{args['file_extension']}"),
                           root_dir=args["data_dir"], recursive=True)

    post_images = glob.glob(os.path.join(args["post_dir"], args["image_dir"], f"*{args['file_extension']}"),
                            root_dir=args["data_dir"], recursive=True)
    post_labels = glob.glob(os.path.join(args["post_dir"], args["label_dir"], f"*{args['file_extension']}"),
                            root_dir=args["data_dir"], recursive=True)
    
    pre_list = [
        {
            args["image_key"]: pre_image,
            args["label_key"]: pre_label
        }
        for pre_image, pre_label in zip(pre_images, pre_labels)
    ]

    post_list = [
        {
            args["image_key"]: post_image,
            args["label_key"]: post_label
        }
        for post_image, post_label in zip(post_images, post_labels)
    ]
    
    return pre_list + post_list


def main(args):
    if args["csv_file"] is None:
        print("Creating JSON from folder search")
        data = from_dir(args)
    else:
        print(f"Creating JSON from {args['csv_file']}")
        data = from_csv(args)

    json_pth = os.path.join(args["data_dir"], args["json_file"])

    data_num = len(data)
    train_num = args["num_train"]
    val_num = args["num_val"]

    if args["num_test"] is None:
        print("Validation number not specified")
        test_num = data_num - train_num - val_num
        if test_num < 0:
            raise ValueError(f"Training and validation sets too large for dataset size {data_num}")

        print(f"Setting testing set size to {test_num}")
    else:
        test_num = args["num_test"]

    if args["shuffle"]:
        random.seed(0)
        random.shuffle(data)

    train_data = data[:train_num]
    val_data = data[train_num:train_num + val_num]
    test_data = data[train_num + val_num:train_num + val_num + test_num]

    total_items = len(train_data) + len(val_data) + len(test_data)
    if total_items != data_num:
        print(f"{total_items}/{data_num} data items used")

    json_dict = dict()
    json_dict["description"] = args["data_name"]

    json_dict["numTrain"] = len(train_data)
    json_dict["numValidate"] = len(val_data)
    json_dict["numTest"] = len(test_data)

    json_dict["training"] = train_data
    json_dict["validation"] = val_data
    json_dict["test"] = test_data

    with open(json_pth, "w+") as json_file:
        print(f"Saving data to {json_pth}")
        json.dump(json_dict, json_file, indent=2)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(vars(parsed_args))
