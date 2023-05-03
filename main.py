import cv2
import os
from locate import locate_plate
from rectify import rectify_plate
from segment import segment_chars
from recognize import recognize_licenses
from evaluate import compute_accuracy, compute_char_jac_sim, compute_2gram_jac_sim


res = []
for root, ds, fs in os.walk("resources/images"):
    for f in fs:
        path = os.path.join(root, f)
        img = cv2.imread(path)
        if f[0] == '1':
            chars = segment_chars(img, visualize=False, easy=True)
        else:
            plate, corners = locate_plate(img, visualize=False)
            plate = rectify_plate(plate, corners, visualize=False)
            chars = segment_chars(plate, visualize=False)
        res.append([f, chars])

recognize_licenses(res)

print('-'*25 + "Results" + '-'*25)
for f, license in res:
    print(f"{f}: {license[:2]+'·'+license[2:]}")

print('-'*25 + "Evaluation" + '-'*25)
compute_accuracy(res)
compute_char_jac_sim(res)
compute_2gram_jac_sim(res)
