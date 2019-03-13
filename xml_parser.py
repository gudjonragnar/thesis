from bs4 import BeautifulSoup as Soup
import numpy as np

def soupify_xml(filename):
    with open(filename) as f:
        soup =  Soup(f,'xml')
    return soup

def process_object(obj, img_name):
    box = obj.bndbox
    left_top = np.array([int(box.xmin.get_text()), int(box.ymin.get_text())])
    right_bot = np.array([int(box.xmax.get_text()), int(box.ymax.get_text())])
    center = 0.5*(left_top + right_bot)

    class_name = obj.find_all('name')[0].get_text()
    # Here we need to have the classes such that we can give them id's
    # Currently outputs class_name instead of class id
    # Output is [path, tuple(center), class_name]?
    return [img_name, tuple(center), class_name]

def process_soup(soup):
    folder = soup.find_all('folder')[0].get_text()
    filename = soup.find_all('filename')[0].get_text()
    path = soup.find_all('path')[0].get_text()
    objects = soup.find_all('object')
    centers = []
    for o in objects:
        centers.append(process_object(o, path))


soup = soupify_xml('../CRCHistoPhenotypes_2016_04_28/Classification/img4/img4.xml')