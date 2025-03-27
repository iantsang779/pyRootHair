import matplotlib.pyplot as plt
from PIL import Image
import os

img_list = ['maize_3_1_stitch.png','maize_3_2_stitch.png', 'maize_3_3_stitch.png', 'maize_3_4_stitch.png']
img_path = '/home/iantsang/Images/Maize/Training/Raw'


imgs = [Image.open(os.path.join(img_path, i)) for i in img_list]
min_img_width = min(i.width for i in imgs)

total_height = 0

for i, img in enumerate(imgs):
    if img.width > min_img_width:
        print('HELMET')
        imgs[i] = img.resize(min_img_width, int(img.height / img.width * min_img_width), Image.ANTIALIAS)
        total_height += imgs[i].height
    else:
        total_height = imgs[i].height * len(img_list)
img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))

y = 0

for img in imgs:
    img_merge.paste(img, (0,y))
    print(y)
    y += img.height

# plt.imshow(img_merge)
img_merge.show()