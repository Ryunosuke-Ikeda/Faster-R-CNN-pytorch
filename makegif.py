
from PIL import Image
from glob import glob
from natsort import natsorted
import os

def make_gif(path):

    #INPUT_DIR='../output'
    #image_files = natsorted(glob(os.path.join(INPUT_DIR, 'BO10/*.jpg')))
    image_files = natsorted(glob(os.path.join(path, '*.jpg')))
    images = []

    for file_name in image_files:

        image = Image.open(file_name)
        t_scale_tate=400  ##目標のスケール(縦)
        #縮小比を計算
        ratio=t_scale_tate/image.size[1]
        ##目標横スケールを計算
        t_scale_yoko=image.size[0]*ratio
        t_scale_yoko=int(t_scale_yoko)
        #リサイズ
        image = image.resize((t_scale_yoko,t_scale_tate)) 
        
        im = image.resize((t_scale_yoko,t_scale_tate))
        images.append(im)

    images[0].save('{}/out.gif'.format(path), save_all=True, append_images=images[1:], loop=0, duration=140)

#make_gif('./output/BOpO13')