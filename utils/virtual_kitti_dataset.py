import os
import regex as re
import math
import csv
import random
from typing import List, Dict


def write_csv(file_path, data: List[Dict],
              fieldnames=['rgb_file', 'depth_file']):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def write_csv_kitti_dataset():
    root_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../..")
        )
    base_path = os.path.abspath(
        os.path.join(root_dir, 'data', "virtual_kitti_2")
        )
    depth_dir = os.path.join(base_path, 'vkitti_2.0.3_depth')
    rgb_dir = os.path.join(base_path, 'vkitti_2.0.3_rgb')
    test_csv_path = os.path.join(base_path, 'splits', 'test.csv')
    train_csv_path = os.path.join(base_path, 'splits', 'train.csv')
    val_csv_path = os.path.join(base_path, 'splits', 'val.csv')
    all_data = []

    for scene in os.listdir(rgb_dir):
        scene_dir = os.path.join(rgb_dir, scene)
        for view in os.listdir(scene_dir):
            rgb_camera_dir = os.path.join(scene_dir, view, 'frames', 'rgb')
            depth_camera_dir = os.path.join(depth_dir, scene, view,
                                            'frames',
                                            'depth')
            for camera in os.listdir(rgb_camera_dir):
                rgb_image_dir = os.path.join(rgb_camera_dir, camera)
                depth_image_dir = os.path.join(depth_camera_dir, camera)
                for image_file in os.listdir(rgb_image_dir):
                    rgb_file_path = os.path.join(rgb_image_dir, image_file)
                    frame_number = re.match(r'rgb_([0-9]{5}).jpg',
                                            image_file).group(1)
                    depth_file = f'depth_{frame_number}.png'
                    depth_file_path = os.path.join(depth_image_dir, depth_file)
                    rgb_rel_path = os.path.relpath(rgb_file_path, root_dir)
                    depth_rel_path = os.path.relpath(depth_file_path, root_dir)
                    data = {"rgb_file": rgb_rel_path,
                            "depth_file": depth_rel_path}
                    all_data.append(data)
    random.shuffle(all_data)
    test_index = math.floor(0.2 * len(all_data))
    train_index = math.floor(0.7 * len(all_data)) + test_index
    train_data = all_data[:train_index]
    test_data = all_data[test_index:train_index]
    val_data = all_data[train_index:]

    write_csv(test_csv_path, test_data)
    write_csv(train_csv_path, train_data)
    write_csv(val_csv_path, val_data)


def main():
    write_csv_kitti_dataset()


if __name__ == "__main__":
    main()
