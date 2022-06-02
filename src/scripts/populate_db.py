import numpy as np
import os
import matplotlib.pyplot as plt

# accessing the raw data
curr_file_path = os.path.dirname(__file__)
path_to_raw = os.path.join(curr_file_path, os.pardir, os.pardir, 'data', 'raw')
path_to_raw = path_to_raw + os.sep
print(path_to_raw)


    
f = open(path_to_raw + 'train-images-idx3-ubyte', 'r')


image_size = 28
num_images = 5
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
image = np.asarray(data[1]).squeeze()
plt.imshow(image)
plt.show()