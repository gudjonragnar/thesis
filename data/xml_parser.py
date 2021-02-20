from os.path import split
from bs4 import BeautifulSoup as Soup
import numpy as np
from PIL import Image
import os
from params import sccnn_params as params

"""
A script to extract centers from xml for labeled data. The xml is the output of LabelIMG.
"""
classes = ["other", "fibroblast", "epithelial", "inflammatory"]
class_map = {
    "fibroblast and endothelial": "fibroblast",
    "lymphocyte": "inflammatory",
    "epithelial": "epithelial",
    "apoptosis / civiatte body": "other",
}
class_to_id = dict(zip(classes, range(len(classes))))


def soupify_xml(filename: str) -> Soup:
    with open(filename, "r") as f:
        soup = Soup(f, "xml")
    return soup


def process_object(obj, img_name):
    box = obj.bndbox
    left_top = np.array([int(box.xmin.get_text()), int(box.ymin.get_text())])
    right_bot = np.array([int(box.xmax.get_text()), int(box.ymax.get_text())])
    center = 0.5 * (left_top + right_bot)

    class_name = obj.find_all("name")[0].get_text()
    # Here we need to have the classes such that we can give them id's
    # Currently outputs class_name instead of class id
    # Output is [path, x, y, class_name]
    return [img_name, center[0], center[1], class_name]


def split_img(path, pos):
    # We split the image since it takes long to load the big image everytime during training
    img = Image.open(path)
    for i in range(len(pos)):
        pane = img.crop(pos[i])
        pane.save("{}_{}.tif".format(path[:-4], i))


def process_soup(soup: Soup, path: str):
    objects = soup.find_all("object")
    centers = []
    for obj in objects:
        processed = process_object(obj, path)
        # Standardize classes and map to id
        if processed[-1] in class_map:
            processed[-1] = class_map[processed[-1]]
        else:
            processed[-1] = "other"
        processed[-1] = class_to_id[processed[-1]]
        centers.append(processed)
    return centers


def _extend_pane(pane, extra, img_size):
    """This will extend the borders of the image if the extension is within the original image
    i.e not outside the border.
    """
    return np.concatenate(
        (
            np.maximum(pane[:2] - extra, [0, 0]),
            np.minimum(pane[2:] + extra, [img_size, img_size]),
        )
    )


def postprocess_centers(centers, panes, shitfted_panes):
    """This function goes through all centers and fixes the coordinates to fit with the split image"""
    true_array = np.array([True, True, False, False])
    for i, center in enumerate(centers):
        x, y = center[1:3]
        tmp_c = np.array([x, y, x, y])
        # Deciding which pane the item should belong to.
        # We choose the left and uppermost pane that contains the center.
        pane_num = [
            j
            for j in range(len(panes))
            if np.array_equal(tmp_c >= panes[j], true_array)
        ][0]
        path = "{}_{}.tif".format(center[0][:-4], pane_num)
        centers[i][0] = path
        centers[i][1:3] = center[1:3] - shitfted_panes[pane_num][:2]
    return centers


def split_by_percentage(centers, p):
    centers_by_class = [
        [cent for cent in centers if cent[3] == i] for i in range(params.num_classes)
    ]
    train_out = []
    test_out = []
    for c in centers_by_class:
        num_of_train = int(np.ceil(len(c) * p))
        train_out.extend(c[:num_of_train])
        test_out.extend(c[num_of_train:])
    return train_out, test_out


if __name__ == "__main__":
    root_dir = params.root_dir
    split_percentage = 0.8
    all_centers = []
    img_size = 2000
    pane_size = 500
    extra = 50
    num_panes = img_size // pane_size
    # splitting the image into panes
    panes = [
        np.array([l, t, l + 1, t + 1]) * pane_size
        for l in range(num_panes)
        for t in range(num_panes)
    ]
    # padding each pane since cells could be on the border of the pane
    extended_panes = [_extend_pane(pane, extra, img_size) for pane in panes]

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root, file)
                soup = soupify_xml(path)
                img_path = path.replace(".xml", ".tif")
                split_img(img_path, extended_panes)
                centers = process_soup(soup, img_path)
                all_centers.extend(centers)
                print(
                    f"File: {path}\n\tnumber of centers: {len(centers)}\n\ttotal            : {len(all_centers)}"
                )

    all_centers = postprocess_centers(all_centers, panes, extended_panes)
    np.random.shuffle(all_centers)

    train, test = split_by_percentage(all_centers, split_percentage)
    print(
        f"number of train: {len(train)}, number of test: {len(test)}. Sum up to: {len(train) + len(test)}"
    )
    class_count = {
        i: len([x for x in train if x[3] == i]) for i in range(params.num_classes)
    }
    class_count["total"] = len(train)

    np.save(os.path.join(root_dir, "train_list.npy"), train)
    np.save(os.path.join(root_dir, "test_list.npy"), test)
    np.save(os.path.join(root_dir, "class_weights.npy"), class_count)
