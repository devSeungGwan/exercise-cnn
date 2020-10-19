import sys
import os
from PIL import Image

START = 600
END = 250
WIDTH = 1400
HEIGHT = 500
FOLDER_NAME = "preprocessing"

# 폴더 생성
def makeFolder(name):
    try:
        if not(os.path.isdir(name)):
            os.makedirs(os.path.join(name))
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            print("같은 이름의 폴더가 존재합니다. 새로운 폴더명을 적어주세요.")
            raise

# 이미지 크롭
def cropImg(path):
    imgs = sorted(list(list(os.walk("{}/block/{}".format(sys.path[0], path)))[0])[2])
    makeFolder("./{}/{}".format(FOLDER_NAME,path))
    for img in imgs:
        im = Image.open(os.path.join("./block", path, img))

        cropImg = im.crop((
            START, 
            END, 
            START+WIDTH, 
            END+HEIGHT
        ))
        cropImg.save("./{}/{}/{}".format(FOLDER_NAME, path, img))

def preprocessing():
    makeFolder(FOLDER_NAME) # 이미지를 저장할 폴더 생성
    category = sorted(list(os.walk("{}/block".format(sys.path[0])))[0][1]) # block 내 폴더 검색

    for path in category: cropImg(path)

if __name__ == "__main__":
    preprocessing()