import cv2
import glob
import copy

corrupted_counter = 0
for i in glob.glob("masks/*.png"):
    print("working on " + i)
    img = cv2.imread(i)
    cimg = copy.deepcopy(img)
    corrupt = []
    for g in range(len(img)):
        for j in range(len(img[g])):
            for c in range(len(img[g][j])):
                if img[g][j][c] > 1 or img[g][j][c] < 0:
                    corrupt.append([g, j, c, img[g][j][c]])
    if len(corrupt) > 0:
        print("corrupt found!")
        corrupted_counter += 1
        max_crr = 0
        for g in corrupt:
            cimg[g[0]][g[1]][g[2]] = 255
            if g[3] > max_crr:
                max_crr = g[3]
        print("maximal corruption: " + str(max_crr))
        print("number of corrupted pixels: " + str(len(corrupt) / 3))
        # cv2.imshow("corrupted image", img)
        # cv2.imshow("corrupted image hsv", cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_RGB2HLS))
        # cv2.imshow("corruption", cimg)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

print("number of viewed images: " + str(len(glob.glob("masks/*.png"))))
print("number of corrupted images: " + str(corrupted_counter))
print("done!")