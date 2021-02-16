from bs4 import BeautifulSoup as Soup
import numpy as np
import os

"""
A script to extract centers from xml for labeled data
"""


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
    # Output is [path, tuple(center), class_name]?
    return [img_name, tuple(center), class_name]


def process_soup(soup: Soup, path: str):
    objects = soup.find_all("object")
    centers = []
    for obj in objects:
        centers.append(process_object(obj, path))
    return centers


if __name__ == "__main__":
    root_dir = "/Users/gudjonragnar/Documents/KTH/Thesis/DataLabeled"
    all_centers = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root, file)
                soup = soupify_xml(path)
                centers = process_soup(soup, path)
                all_centers.extend(centers)

    print(f"Found {len(all_centers)} centers")
    np.save(os.path.join(root_dir, "all_centers.npy"), all_centers)
