from PIL import Image
import math
import os, sys

path = "/home/student/Project/Bar Channelwise-RGB Merged(Horizontal)/data/tactile/"
dirs = os.listdir( path )

def make_square(im, min_size=256, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = 256
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resize():
    for item in dirs:
        if os.path.isfile(path+item) and "-r" not in path+item:
            im = Image.open(path+item)
            x,y = im.size
            f, e = os.path.splitext(path+item)
            if x == y:
                
                imResize = im.resize((256,256), Image.ANTIALIAS)
                #imResize.save(f + '.tiff', 'TIFF', quality=90)
                #imResize = imResize.convert('RGBA')
                #f,e = os.path.splitext(destPath+item)
                print(f+'.tiff')
                imResize.save(f + '.tiff','TIFF')
                print(f)
                print(e)
            else:
            	if x>y:
            	    ratio = 256/x
            	    x = 256
            	    y = math.floor(y*ratio)
            	else:
            	    ratio = 256/y
            	    y = 256
            	    x = math.floor(x*ratio)
            	im = im.resize((x,y), Image.ANTIALIAS)
            	imResize = make_square(im)
            	imResize.save(f + '.tiff','TIFF')
                
        
resize()
