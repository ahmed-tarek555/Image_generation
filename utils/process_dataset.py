import os
import torch
from process_img import process_img

def preprocess_and_save(raw_dir, save_dir):
    os.chdir("..")
    os.makedirs(save_dir, exist_ok=True)

    for person in os.listdir(raw_dir):
        person_dir = os.path.join(raw_dir, person)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name) # -> faces/ahmed/img
            processed_img = process_img(img_path)
            save_path = os.path.join(save_dir, img_name.split('.')[0] + ".pt")
            torch.save(processed_img, save_path)
    print('Done')

preprocess_and_save('faces', 'real')