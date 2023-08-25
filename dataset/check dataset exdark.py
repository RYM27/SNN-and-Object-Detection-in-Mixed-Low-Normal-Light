import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

dataset = fo.Dataset("ExDark")
for split in ["train", "test", "val"]:
    dataset.add_dir(
                # please change the dataset dir to your directory
                dataset_dir="D:\\Data RYM\\Data Kuliah\\S2\\Semester 1\\Thesis\\Percobaan\\Dataset\\ExDark\\all",
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                tags=split,
    )

if __name__ == "__main__":
# Visualize the dataset in the FiftyOne App
    session = fo.launch_app(dataset)
    session.wait()