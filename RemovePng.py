#importing os module
import os
#path of the folder
folder_path = (r'/home/student/Project/.../data/tactile/')
#using listdir() method to list the files of the folder
test = os.listdir(folder_path)
#taking a loop to remove all the images
for images in test:
    if images.endswith(".png"):   #remove all ".png" images from mentioned folder_path
        os.remove(os.path.join(folder_path, images))
