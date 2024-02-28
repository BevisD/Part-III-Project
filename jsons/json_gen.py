import json
import os
import csv
import random
import argparse


# Parse the input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="JSON file creator from csv file")
    parser.add_argument("--csv-file", default="labels.csv", type=str, help="csv file containing patient data")
    parser.add_argument("--json-file", default="data.json", type=str, help="json file to store patient data")
    parser.add_argument("--data-dir", default="./", type=str, help="json file to store patient data")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the data or not")
    parser.add_argument("--num-train", type=int, help="number of training datapoints", required=True)
    parser.add_argument("--num-test", type=int, help="number of testing datapoints", required=True)
    parser.add_argument("--num-val", type=int, help="number of validation datapoints", required=True)
    parser.add_argument("--data-name", default="neov", help="name of the dataset")

    args = parser.parse_args()
    return args


def main(args):

    csv_pth = os.path.join(args["data_dir"], args["csv_file"])
    json_pth = os.path.join(args["data_dir"], args["json_file"])

    train_num = args["num_train"]
    test_num = args["num_test"]
    val_num = args["num_val"]

    json_dict = {}
    json_dict["description"] = args["data_name"]

    patients = []

    with open(csv_pth, "r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

        for row in reader:
            info = dict(zip(header, row))

            pre_filename = info["pre_treatment"]
            post_filename = info["post_treatment"]
            recist_label = info["RECIST_label"]

            # Assumes there are pre_treatment and post_treatment folders
            # Each folder has images and segmentations
            pre_img_path = os.path.join("pre_treatment/images", pre_filename)
            pre_seg_path = os.path.join("pre_treatment/segmentations", pre_filename)

            post_img_path = os.path.join("post_treatment/images", post_filename)
            post_seg_path = os.path.join("post_treatment/segmentations", post_filename)

            patients.append({
                "pre-image": pre_img_path,
                "pre-label": pre_seg_path,
                "post-image": post_img_path,
                "post-label": post_seg_path,
                "recist-label": recist_label
            })

    if args["shuffle"]:
        random.seed(0)
        random.shuffle(patients)

    train = patients[:train_num]
    test = patients[train_num:train_num + test_num]
    val = patients[train_num + test_num:train_num + test_num + val_num]

    json_dict["numTrain"] = len(train)
    json_dict["numTest"] = len(test)
    json_dict["numValidate"] = len(val)

    json_dict["training"] = train
    json_dict["test"] = test
    json_dict["validation"] = val

    with open(json_pth, "w+") as json_file:
        json.dump(json_dict, json_file, indent=2)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(vars(parsed_args))
