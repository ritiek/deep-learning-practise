from PIL import Image
import numpy as np

img = Image.open('/home/ritiek/Pictures/id_card.jpg')

# optionally convert to grayscale
# img = img.convert('L')
data = np.asarray(img, dtype='int32')

print(data)
