import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

path = "/home/n001/Documents/Pedro_Antonio/animationgif_g/"

for m in range(0,30,1):
    #Read the two images
    image1 = Image.open(path+f'g_{m}.jpg')
    #image1.show()
    image2 = Image.open(path+f't_{m}.jpeg')
    #image2.show()
    #resize, first image
    image1 = image1.resize((960, 720))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250), dpi=(40))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(path+f"/merge/merged{m}.jpg","JPEG")