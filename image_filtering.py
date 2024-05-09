import numpy as np
import os
import colorsys
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from sklearn.cluster import KMeans
import cv2
from collections import Counter
import shutil
import datetime

# load sample images
def load_images(path_img):
    dim = (256, 256)

    images = {}

    for img_name in os.listdir(path_img):
        img_path = os.path.join(path_img, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        images[img_name] = img

    return images

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

def save_img(img, img_name, dest_dir):
    dest_path = dest_dir + img_name
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest_path, img)

clt_3 = KMeans(n_clusters=3, n_init=10)
clt_2 = KMeans(n_clusters=2, n_init=10)

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step : int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette, perc, k_cluster.cluster_centers_

threshold_lum_low_1 = 90; threshold_lum_high_1 = 175
threshold_delta_e_low_2 = 14; threshold_max_lum_low_2 = 115; threshold_max_lum_high_2 = 215; threshold_perc_max_lum_low_2 = 0.28; threshold_perc_max_lum_high_2 = 0.8; threshold_perc_max_2 = 0.82
threshold_hue_min_lum_low_3 = 250; threshold_hue_min_lum_high_3 = 290; threshold_perc_min_lum_low_3 = 0.16; threshold_perc_min_lum_high_3 = 0.46; threshold_lum_min_3 = 120; threshold_perc_min_3 = 0.09; threshold_perc_max_3 = 0.6
threshold_sobel_edge_detection = 50; threshold_blob_detection = 10; threshold_blur_detection = 200

customise = int(input("Chcete nastaviť svoje vlastné hraničné hodnoty? nie [0], áno [1] : "))
if customise == 1:
    threshold_lum_low_1 = int(input("Zadajte dolnú hranicu prípustného jasu priemernej farby (90): "))
    threshold_lum_high_1 = int(input("Zadajte hornú hranicu prípustného jasu priemernej farby (175): "))

    threshold_delta_e_low_2 = int(input("Zadajte dolnú hranicu prípustného farebného rozdielu CIE2000 (14): "))
    threshold_max_lum_low_2 = int(input("Zadajte dolnú hranicu prípustného jasu svetlejšieho z 2 zhlukov (115): "))
    threshold_max_lum_high_2 = int(input("Zadajte hornú hranicu prípustného jasu svetlejšieho z 2 zhlukov (215): "))
    threshold_perc_max_lum_low_2 = float(input("Zadajte dolnú hranicu prípustného pomeru svetlejšieho z 2 zhlukov (0.28): "))
    threshold_perc_max_lum_high_2 = float(input("Zadajte hornú hranicu prípustného pomeru svetlejšieho z 2 zhlukov (0.8): "))
    threshold_perc_max_2 = float(input("Zadajte hornú hranicu prípustného pomeru väčšieho z 2 zhlukov (0.82): "))

    threshold_hue_min_lum_low_3 = int(input("Zadajte dolnú hranicu prípustného odtieňa najtmavšieho z 3 zhlukov (250): "))
    threshold_hue_min_lum_high_3 = int(input("Zadajte hornú hranicu prípustného odtieňa najtmavšieho z 3 zhlukov (290): "))
    threshold_perc_min_lum_low_3 = float(input("Zadajte dolnú hranicu prípustného pomeru najtmavšieho z 3 zhlukov (0.16): "))
    threshold_perc_min_lum_high_3 = float(input("Zadajte hornú hranicu prípustného pomeru najtmavšieho z 3 zhlukov (0.46): "))
    threshold_lum_min_3 = int(input("Zadajte hornú hranicu prípustného jasu najtmavšieho z 3 zhlukov (120): "))
    threshold_perc_min_3 = float(input("Zadajte dolnú hranicu prípustného pomeru najmenšieho z 3 zhlukov (0.09): "))
    threshold_perc_max_3 = float(input("Zadajte hornú hranicu prípustného pomeru najväčšieho z 3 zhlukov (0.6): "))

    threshold_sobel_edge_detection = int(input("Zadajte dolnú hranicu prípustnej priemernej farby čiernobielej snimky hrán (50): "))
    threshold_blob_detection = int(input("Zadajte dolnú hranicu prípustného počtu kruhovitých fľakov získaného metódou Simple Blob Detector (10): "))
    threshold_blur_detection = int(input("Zadajte dolnú hranicu prípustnej miery zaostrenia snímky získanej prechodom cez Laplaceov filter (200): "))

# https://github.com/mrakelinggar/data-stuffs/tree/master/frequent_color
def clustering_1(images,
                 path_excluded,
                 path_excluded_1_luminance=None,
                 path_excluded_all_clustering=None):

    counter_excluded_1_luminance = 0
    counter_excluded_all = 0

    images_included = {}

    for img_name, img in images.items():
        excluded = False

        # --- 1 cluster = average image color -> luminance ---
        img_temp = img.copy()
        img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = np.average(img, axis=(0, 1))
        dom_color = img_temp[0, 0]

        luminance = 0.2126 * dom_color[0] + 0.7152 * dom_color[1] + 0.0722 * dom_color[2]

        # brightness too low/high
        if luminance <= threshold_lum_low_1 or luminance >= threshold_lum_high_1: # 90-175
            if path_excluded_1_luminance is not None: save_img(img, img_name, path_excluded_1_luminance + '/')
            counter_excluded_1_luminance += 1
            excluded = True

        if excluded:
            if path_excluded_all_clustering is not None: save_img(img, img_name, path_excluded_all_clustering + '/')
            save_img(img, img_name, path_excluded + '/')
            counter_excluded_all += 1
        else:
            images_included[img_name] = img

    print('Vylúčené - jas 1: ', counter_excluded_1_luminance)

    return images_included

def clustering_2(images,
                 path_excluded,
                 path_excluded_2_cie2000=None,
                 path_excluded_2_bright=None,
                 path_excluded_2_perc=None,
                 path_excluded_all_clustering=None):
    
    counter_excluded_2_cie2000 = 0
    counter_excluded_2_bright = 0
    counter_excluded_2_perc = 0
    counter_excluded_all = 0

    images_included = {}

    counter = 0
    for img_name, img in images.items():
        excluded = False

        # 2 K-Means clustering
        clt = clt_2.fit(img.reshape(-1, 3))
        palette, perc, centers = palette_perc(clt)

        centers = np.uint8(centers)

        # 2 means: delta CIE2000
        col1_lab = convert_color(sRGBColor(centers[0][0], centers[0][1], centers[0][2], is_upscaled=True), LabColor)
        col2_lab = convert_color(sRGBColor(centers[1][0], centers[1][1], centers[1][2], is_upscaled=True), LabColor)
        delta_e = delta_e_cie2000(col1_lab, col2_lab)

        # difference between 2 main colours too low
        if delta_e <= threshold_delta_e_low_2:
            if path_excluded_2_cie2000 is not None: save_img(img, img_name, path_excluded_2_cie2000 + '/')
            counter_excluded_2_cie2000 += 1
            excluded = True

        max_perc_2 = max(perc.values())

        # 2 means percentage too high
        if max_perc_2 >= threshold_perc_max_2: # 0.82
            if path_excluded_2_perc is not None: save_img(img, img_name, path_excluded_2_perc + '/')
            counter_excluded_2_perc += 1
            excluded = True

        luminance_2_tmp = []

        for i in range(2):
            luminance = 0.2126 * centers[i][0] + 0.7152 * centers[i][1] + 0.0722 * centers[i][2]
            luminance_2_tmp.append(luminance)

        max_luminance_2 = max(luminance_2_tmp)
        i = luminance_2_tmp.index(max(luminance_2_tmp))
        perc_max_luminance_2 = perc.get(i)

        if not (max_luminance_2 > threshold_max_lum_low_2 and max_luminance_2 < threshold_max_lum_high_2 and # 115-215 (210??)
            perc_max_luminance_2 > threshold_perc_max_lum_low_2 and perc_max_luminance_2 < threshold_perc_max_lum_high_2): # 0.28-0.8???
            if path_excluded_2_bright is not None: save_img(img, img_name, path_excluded_2_bright + '/')
            counter_excluded_2_bright += 1
            excluded = True

        if excluded:
            if path_excluded_all_clustering is not None: save_img(img, img_name, path_excluded_all_clustering + '/')
            save_img(img, img_name, path_excluded + '/')
            counter_excluded_all += 1
        else:
            images_included[img_name] = img

        counter += 1
        if counter % 10000 == 0:
            print('%d images processed' % counter)

    print('Vylúčené - farebný rozdiel 2: ', counter_excluded_2_cie2000)
    print('Vylúčené - svetlé 2: ', counter_excluded_2_bright)
    print('Vylúčené - pomer 2: ', counter_excluded_2_perc)

    return images_included

def clustering_3(images,
                 path_excluded,
                 path_excluded_3_nuclei=None,
                 path_excluded_3_perc=None,
                 path_excluded_all_clustering=None):

    counter_excluded_3_nuclei = 0
    counter_excluded_3_perc = 0
    counter_excluded_all = 0

    images_included = {}

    counter = 0
    for img_name, img in images.items():
        excluded = False

        # 3 K-Means clustering
        clt = clt_3.fit(img.reshape(-1, 3))
        palette2, perc, centers = palette_perc(clt)

        min_perc_3 = min(perc.values())
        max_perc_3 = max(perc.values())

        # 3 means percentage too low/high
        if min_perc_3 <= threshold_perc_min_3 or max_perc_3 >= threshold_perc_max_3: # 0.09-0.6
            if path_excluded_3_perc is not None: save_img(img, img_name, path_excluded_3_perc + '/')
            counter_excluded_3_perc += 1
            excluded = True

        luminance_tmp = []

        for i in range(3):
            luminance = 0.2126 * centers[i][0] + 0.7152 * centers[i][1] + 0.0722 * centers[i][2]
            luminance_tmp.append(luminance)

        min_luminance_3 = min(luminance_tmp)
        i = luminance_tmp.index(min(luminance_tmp))
        min_lum_3_color = centers[i][0], centers[i][1], centers[i][2]

        red, green, blue = min_lum_3_color[0], min_lum_3_color[1], min_lum_3_color[2]
        red, green, blue = red / 255.0, green / 255.0, blue / 255.0
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        hue_min_luminance_3 = hue * 360

        perc_min_luminance_3 = perc.get(i)

        # bad proportion of nuclei
        if not (hue_min_luminance_3 > threshold_hue_min_lum_low_3 and hue_min_luminance_3 < threshold_hue_min_lum_high_3
                and perc_min_luminance_3 > threshold_perc_min_lum_low_3 and perc_min_luminance_3 < threshold_perc_min_lum_high_3
                and min_luminance_3 < threshold_lum_min_3):
            if path_excluded_3_nuclei is not None: save_img(img, img_name, path_excluded_3_nuclei + '/')
            counter_excluded_3_nuclei += 1
            excluded = True

        if excluded:
            if path_excluded_all_clustering is not None: save_img(img, img_name, path_excluded_all_clustering + '/')
            save_img(img, img_name, path_excluded + '/')
            counter_excluded_all += 1
        else:
            images_included[img_name] = img

        counter += 1
        if counter % 10000 == 0:
            print('%d images processed' % counter)

    print('Vylúčené - jadrá 3: ', counter_excluded_3_nuclei)
    print('Vylúčené - pomer 3: ', counter_excluded_3_perc)

    return images_included

# https://hackthedeveloper.com/edge-detection-opencv-python/
def sobel_edge_detection(images_types,
                         path_excluded,
                         path_excluded_sobel=None):

    for images in images_types:

        counter_excluded_mean_edges = 0
        images_included = {}

        for img_name, img in images.items():
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)

            sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

            mag, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
            edges2 = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)[1]

            arr = np.asarray(edges2)
            mean = arr.mean(0).mean(0)

            if mean <= threshold_sobel_edge_detection:
                if path_excluded_sobel is not None: save_img(img, img_name, path_excluded_sobel + '/')
                save_img(img, img_name, path_excluded + '/')
                counter_excluded_mean_edges += 1
            else:
                images_included[img_name] = img

    print('Vylúčené - hrany: ', counter_excluded_mean_edges)

    return images_included

#https://hackthedeveloper.com/blob-detection-opencv-python/
def blob_detection(images_types,
                   path_excluded,
                   path_excluded_blob=None):

    for images in images_types:

        counter_excluded_num_circles = 0
        images_included = {}

        for img_name, img in images.items():
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 1
            params.maxThreshold = 200

            params.filterByArea = True
            params.minArea = 20
            params.maxArea = 1000

            params.filterByCircularity = True
            params.minCircularity = 0.5
            params.maxCircularity = 1

            params.filterByConvexity = True
            params.minConvexity = 0.85
            params.maxConvexity = 1

            detector = cv2.SimpleBlobDetector_create(params)

            keypoints = detector.detect(gray)

            if len(keypoints) <= threshold_blob_detection:
                if path_excluded_blob is not None: save_img(img, img_name, path_excluded_blob + '/')
                save_img(img, img_name, path_excluded + '/')
                counter_excluded_num_circles += 1
            else:
                images_included[img_name] = img

    print('Vylúčené - kruhovité fľaky: ', counter_excluded_num_circles)

    return images_included

def blur_detection(images_types,
                   path_excluded,
                   path_excluded_blur=None):

    for images in images_types:

        counter_excluded_lapl_var = 0
        images_included = {}

        for img_name, img in images.items():
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            lapl_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if lapl_var <= threshold_blur_detection:
                if path_excluded_blur is not None: save_img(img, img_name, path_excluded_blur + '/')
                save_img(img, img_name, path_excluded + '/')
                counter_excluded_lapl_var += 1
            else:
                images_included[img_name] = img

    print('Vylúčené - rozmazanie: ', counter_excluded_lapl_var)

    return images_included

# === FILTER: MODE 1 ===
# testing the effectivity of every filtering method
# each method receives on input all samples

def filter1_test(path_input_dir, path_output_dir):
    path_samples = path_input_dir
    images = load_images(path_samples)
    path_version = path_output_dir + '/'
    os.makedirs(path_version, exist_ok=True)

    # 1. common colors clustering
    print('\n=== K-means zhlukovanie: najčastejšie farby ===')
    path_type_version = path_version + 'common_colors_clustering/'
    os.makedirs(path_type_version, exist_ok=True)

    path_excluded_1_luminance = path_type_version + 'excluded_1_luminance_' +\
        str(threshold_lum_low_1) + '-' + str(threshold_lum_high_1)
    os.makedirs(path_excluded_1_luminance, exist_ok=True)

    path_excluded_2_cie2000 = path_type_version + 'excluded_2_cie2000_' +\
        str(threshold_delta_e_low_2)
    os.makedirs(path_excluded_2_cie2000, exist_ok=True)

    path_excluded_2_bright = path_type_version + 'excluded_2_bright_' +\
        str(threshold_perc_max_lum_low_2) + '-' + str(threshold_perc_max_lum_high_2) + '_' +\
            str(threshold_max_lum_low_2) + '-' + str(threshold_max_lum_high_2)
    os.makedirs(path_excluded_2_bright, exist_ok=True)

    path_excluded_2_perc = path_type_version + 'excluded_2_perc_' + str(threshold_perc_max_2)
    os.makedirs(path_excluded_2_perc, exist_ok=True)

    path_excluded_3_nuclei = path_type_version + 'excluded_3_nuclei_' +\
        str(threshold_perc_min_lum_low_3) + '-' + str(threshold_perc_min_lum_high_3) + '_' +\
            str(threshold_hue_min_lum_low_3) + '-' + str(threshold_hue_min_lum_high_3) + '_' +\
                str(threshold_lum_min_3)
    os.makedirs(path_excluded_3_nuclei, exist_ok=True)

    path_excluded_3_perc = path_type_version + 'excluded_3_perc_' +\
        str(threshold_perc_min_3) + '-' + str(threshold_perc_max_3)
    os.makedirs(path_excluded_3_perc, exist_ok=True)

    path_excluded_all_clustering = path_type_version + 'excluded_all_clustering'
    os.makedirs(path_excluded_all_clustering, exist_ok=True)

    path_excluded = path_version + 'all_excluded_combined'
    os.makedirs(path_excluded, exist_ok=True)

    # returns images_included = a dictionary of images that this method could not filter
    images_included_clustering_1 = clustering_1(images,
                                                path_excluded,
                                                path_excluded_1_luminance,
                                                path_excluded_all_clustering=path_excluded_all_clustering)

    images_included_clustering_2 = clustering_2(images,
                                                path_excluded,
                                                path_excluded_2_cie2000,
                                                path_excluded_2_bright,
                                                path_excluded_2_perc,
                                                path_excluded_all_clustering=path_excluded_all_clustering)

    images_included_clustering_3 = clustering_3(images,
                                                path_excluded,
                                                path_excluded_3_nuclei,
                                                path_excluded_3_perc,
                                                path_excluded_all_clustering=path_excluded_all_clustering)

    print('Vylúčené - zhlukovanie: ', len(os.listdir(path_excluded_all_clustering)))

    images_types = []
    images_types.append(images)

    # 2. edge detection
    print('\n=== Sobelova detekcia hrán ===')
    path_excluded_sobel = path_version + 'sobel_edge_detection/excluded_mean_edges_' + str(threshold_sobel_edge_detection)
    os.makedirs(path_excluded_sobel, exist_ok=True)
    sobel_edge_detection(images_types, path_excluded, path_excluded_sobel)

    # 3. circle detection
    print('\n=== Detekcia kruhovitých fľakov ===')
    path_excluded_blob = path_version + 'circle_detection/excluded_num_circles_' + str(threshold_blob_detection)
    os.makedirs(path_excluded_blob, exist_ok=True)
    blob_detection(images_types, path_excluded, path_excluded_blob)

    # 4. blur detection
    print('\n=== Laplaceovská detekcia rozmazania ===')
    path_excluded_blur = path_version + 'blur_detection/excluded_lapl_var_' + str(threshold_blur_detection)
    os.makedirs(path_excluded_blur, exist_ok=True)
    blur_detection(images_types, path_excluded, path_excluded_blur)

    print('Vylúčené všetky skombinované: ', len(os.listdir(path_excluded)))
    print('\n')

    return path_excluded

# === FILTER: MODE 2 ===
# gradual filtering
# each method receives on input only those samples that have not yet been filtered

def filter2_clean(path_input_dir, path_output_dir):
    path_samples = path_input_dir
    images = load_images(path_samples)
    path_version = path_output_dir + '/'
    os.makedirs(path_version, exist_ok=True)

    path_excluded = path_version + 'all_excluded_combined'
    os.makedirs(path_excluded, exist_ok=True)

    images_types = []
    images_types.append(images)

    # 1. edge detection
    print('\n=== Sobelova detekcia hrán ===')
    images_included_sobel = sobel_edge_detection(images_types, path_excluded)

    images_types = []
    images_types.append(images_included_sobel)

    # 2. circle detection
    print('\n=== Detekcia kruhovitých fľakov ===')
    images_included_blob = blob_detection(images_types, path_excluded)

    images_types = []
    images_types.append(images_included_blob)

    # 3. blur detection
    print('\n=== Laplaceovská detekcia rozmazania ===')
    images_included_blur = blur_detection(images_types, path_excluded)

    # 4. common colors clustering
    print('\n=== K-means zhlukovanie: najčastejšie farby ===')
    images_included_clustering_1 = clustering_1(images_included_blur,
                                                path_excluded)

    images_included_clustering_2 = clustering_2(images_included_clustering_1,
                                                path_excluded)

    images_included_clustering_3 = clustering_3(images_included_clustering_2,
                                                path_excluded)

    print('Vylúčené všetky skombinované: ', len(os.listdir(path_excluded)))
    print('Ponechané všetky skombinované: ', len(images_included_clustering_3), '\n')
    
    return path_excluded

# we have all excluded images
# the rest of the unfiltered dataset are included images
def copy_files_ignore(source_dir, destination_dir, ignore_dir):
    source_files = os.listdir(source_dir)
    ignore_files = os.listdir(ignore_dir)
    
    for file_name in source_files:
        source_path = os.path.join(source_dir, file_name)
        
        if file_name not in ignore_files:
            destination_path = os.path.join(destination_dir, file_name)
            shutil.copyfile(source_path, destination_path)

def main():
    path_input_dir = input("Zadajte cestu adresára datsetu na čistenie: ")
    while not os.path.exists(path_input_dir) or not os.path.isdir(path_input_dir):
        print("Adresár datasetu na čistenie nebol nájdený!")
        path_input_dir = input("Zadajte cestu adresára datasetu na čistenie: ")

    path_output_dir = input("Zadajte cestu adresára na ukladanie výstupov alebo stlačte Enter pre ukladanie výstupov do lokálneho adresára: ")
    if not path_output_dir:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_output_dir = "filter-output_" + timestamp
        os.makedirs(path_output_dir, exist_ok=True)

    path_included_dir = os.path.join(path_output_dir, "all_included_combined")
    os.makedirs(path_included_dir, exist_ok=True)

    print("Mód 1: Pre testovanie kritérií stlačte [1]")
    print("Mód 2: Pre aplikovanie filtra na vyčistenie datasetu stlačte [2]")
    mode = int(input("Zadajte mód (1/2): "))

    if mode == 1:
        print("Spúšťam filtrovanie v móde 1...")
        excluded = filter1_test(path_input_dir, path_output_dir)
        #copy_files_ignore(path_input_dir, path_included_dir, excluded)
        print("Filtrovanie v móde 1 bolo dokončené.")
    if mode == 2:
        print("Spúšťam filtrovanie v móde 2...")
        excluded = filter2_clean(path_input_dir, path_output_dir)
        #copy_files_ignore(path_input_dir, path_included_dir, excluded)
        print("Filtrovanie v móde 2 bolo dokončené.")

if __name__ == "__main__":
    main()