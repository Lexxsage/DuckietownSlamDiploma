import glob
import cv2

for name in glob.glob('*.png'):
    num = ""
    for s in name:
        if s.isnumeric():
            num += s
    new_name = "img." + num + ".png"
    img = cv2.imread(name)
    print(name, new_name)
    cv2.imwrite(new_name, img)
