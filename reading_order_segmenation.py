import cv2
import numpy as np


def edge_filter(stats):
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    res_y, res_x = image_shape
    return stats[((res_x - w - x) * (res_y - h - y) * x * y) != 0]

def draw_rect(stats):
    for sgmt in stats:
        x, y, w, h = sgmt[:4]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

image = cv2.imread('sample.png')
image_shape = image.shape[:2]
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray, 5)
thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
stats = cv2.connectedComponentsWithStats(thresh, connectivity=8)[2][1:]
filtered = edge_filter(stats)
sorted_top = filtered[np.argsort(filtered[:, 1])]
median_height = np.median(sorted_top[:, 3])
diff_y = np.diff(sorted_top[:,1])
new_line = np.where(diff_y > median_height)[0] + 1
lines = np.array_split(sorted_top, new_line)
sorted_left = [line[np.argsort(line[:, 0])] for line in lines]

for line, num in zip(sorted_left, range(len(sorted_left))):
    print(f'line {num + 1}:\n {line}\n')

draw_rect(filtered)
cv2.imshow("image", image)
cv2.waitKey()