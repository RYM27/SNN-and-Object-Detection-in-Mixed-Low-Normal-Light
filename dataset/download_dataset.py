import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

#
# Only the required images will be downloaded (if necessary).
# By default, only detections are loaded
#

label_list = ("bicycle", 
              "boat",
              "bottle",
              "bus",
              "car",
              "cat",
              "chair",
              "cup",
              "dog",
              "motorcycle",
              "person",
              "dining table"
              )

dataset_name = "all"
"""
DATASET SIZE
train + val = 3,000 + 1,800 = 4,800
test = 2,563
"""
dataset_size = 2563

# download dataset from zoo


"""
SPLITS
train and val -> splits=["train"]
test -> splits=["validation"]
"""
dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["validation"],
    classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog",
            "motorcycle", "person", "dining table" ],
    #classes=[f"{dataset_name}"],
    max_samples=dataset_size,
)

dataset.name = f"coco-2017-train-validation-{dataset_name}-{dataset_size}"


#dataset = fo.load_dataset(f"coco-2017-train-validation-{dataset_name}-{dataset_size}")
dataset.persistent = False
# change it to train or validation based on the dataset splits before
dataset.untag_samples("validation")



# create new view

view = (dataset
        .select_fields("ground_truth")
        .filter_labels("ground_truth", F("label").is_in(label_list))
        .limit(dataset_size)
)

"""
RANDOM SPLIT
train and val -> "train": 0.625, "val": 0.375
test -> "test": 1
"""
four.random_split(view, {"test": 1})
#"train": 0.625, "val": 0.375
#"train": 0.4074426184978949, "val": 0.2444655710987369, "test": 0.3480918104033682
print(view.count_sample_tags())

dataset.save_view(f"{dataset_name} {dataset_size}", view)


if __name__ == "__main__":
# Visualize the dataset in the FiftyOne App
    session = fo.launch_app(dataset)
    session.wait()

    # load the dataset and export it to yolo format
    #view = dataset.load_saved_view(f"{dataset_name} {dataset_size}")
    # please change the export dir according to your directory
    export_dir = f"D:\Data RYM\Data Kuliah\S2\Semester 1\Thesis\Percobaan\Dataset\COCO Normal\{dataset_name}"
    label_field = "ground_truth"

    # change split to the selected split
    #for split in ["train", "val"]:
    for split in ["test"]:
        split_view = view.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=list(label_list),
            split=split
        )
