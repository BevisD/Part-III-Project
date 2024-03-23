import argparse
import json
import os

parser = argparse.ArgumentParser(description="JSON file creator from csv file")
parser.add_argument("--json-out-file", default="data.json", type=str, help="json file to store patient data")
parser.add_argument("--json-in-file", action="append", type=str, help="json file containing patient data", required=True)
parser.add_argument("--data-name", default="dataset", help="name of the dataset")
parser.add_argument("--shuffle", action="store_true", help="shuffle data on merge")
parser.add_argument("--image-key", default="image", type=str)
parser.add_argument("--label-key", default="label", type=str)


def main(args):
    json_out_dict = {
        "description": args.data_name,
        "numTrain": 0,
        "numValidate": 0,
        "numTest": 0,
        "training": [],
        "validation": [],
        "test": []
    }
    for json_in_file in args.json_in_file:
        with open(json_in_file, "r") as fp:
            json_in_dict = json.load(fp)

        json_out_dict["numTrain"] += json_in_dict["numTrain"]
        json_out_dict["numValidate"] += json_in_dict["numValidate"]
        json_out_dict["numTest"] += json_in_dict["numTest"]

        for key in ["training", "validation", "test"]:
            for data in json_in_dict[key]:
                json_out_dict[key].append(
                    {
                        args.image_key: os.path.join(json_in_dict["description"], data[args.image_key]),
                        args.label_key: os.path.join(json_in_dict["description"], data[args.label_key])
                    })

    with open(args.json_out_file, "w+") as fp:
        json.dump(json_out_dict, fp, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
