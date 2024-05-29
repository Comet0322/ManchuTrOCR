import json
import os
import xml.etree.ElementTree as etree

import cv2

with open("transliter.json") as f:
    manjurules = json.load(f)

# build dictionary
manju_dict = {}
for rule in manjurules:
    manju_dict[rule["manju"]] = rule["roman"]


def transliterate(text):
    result = ""
    for ch in text:
        if ch in manju_dict:
            result += manju_dict[ch]

    return result


base_dir = os.path.dirname(os.path.relpath(__file__))
src_img_dir = os.path.join(base_dir, "src_img")
src_label_dir = os.path.join(base_dir, "annotations")
dst_dir = os.path.join(base_dir)
os.makedirs(f"{dst_dir}/images", exist_ok=True)

# search all xml files in src_dir
label_count = 0
for xml_file in sorted(os.listdir(src_label_dir)):
    if xml_file.endswith(".xml"):
        # parse xml file
        with open(os.path.join(src_label_dir, xml_file), "r") as xml_file:
            tree = etree.parse(xml_file)
        root = next(tree.getroot().iter("images"))
        # read image
        image_path = os.path.join(src_img_dir, root.find("image").get("file"))
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path)
        original_image = image.copy()
        # create annotation file
        annotation = []
        # parse xml and crop image
        for image_tag in root.findall("image"):
            for box in image_tag.findall("box"):
                label = box.findall("label")[0].text
                if len(label.strip()) == 0:
                    continue
                # if find non-manchu  character, skip
                if all(ord(c) < 128 for c in label):
                    continue
                label = transliterate(label)
                # top, left, width, height
                top = int(float(box.get("top"))) - 5
                left = int(float(box.get("left"))) - 5
                width = int(float(box.get("width"))) + 10
                height = int(float(box.get("height"))) + 10

                if width <= 5 or height <= 5:
                    continue
                # crop image
                cropped_image = image[top : top + height, left : left + width]
                # create cropped image name
                cropped_image_name = f"{dst_dir}/images/{image_name}_{top}_{left}_{top+height}_{left+width}.jpg"
                cv2.imwrite(cropped_image_name, cropped_image)
                # record cropped image name and annotation
                annotation.append((cropped_image_name, label))

        label_count += len(annotation)
        # write annotation to csv
    with open(f"{dst_dir}/labels.csv", "w", encoding="utf-8") as f:
        f.write("image_id,text\n")
        for cropped_image_name, label in annotation:
            f.write(f"{cropped_image_name},{label}\n")
