import numpy as np
import cv2

def determine_side(im, threshold):
    # set threshold to remove artifacts around edges
    im_thres = im.copy()
    im_thres[im_thres > threshold] = 0

    # determine breast side
    col_sums_split = np.array_split(np.sum(im_thres, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])

    if left_col_sum > right_col_sum:
        breast_side = 'l'
    else:
        breast_side = 'r'
    return breast_side

def segment_breast(image, threshold = 25):
    # remove totally white pixels - artifacts
    img = image.copy()
    img[img==1] = 0

    breast_side = determine_side(img, threshold)
    gray = (img*255).astype(np.uint8)

    # threshold and invert
    thresh1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh1 = 255 - thresh1

    # remove borders
    # count number of white pixels in columns as new 1D array
    count_cols = np.count_nonzero(thresh1, axis=0)
    # crop image
    if breast_side == 'l':
        # get first and last x coordinate where black
        first_x = 0
        last_x = np.where(count_cols>0)[0][-1]
    else:
        # get first and last x coordinate where black
        first_x = np.where(count_cols>0)[0][0]
        last_x = img.shape[0]-1

    # count number of white pixels in rows as new 1D array
    count_rows = np.count_nonzero(thresh1, axis=1)
        # get first and last y coordinate where black
    first_y = max(np.where(count_rows>0)[0][0],5)
    last_y = min(np.where(count_rows>0)[0][-1],img.shape[1]-5)
    crop = img[first_y:last_y+1, first_x:last_x+1]

    # crop thresh1 and invert
    thresh2 = thresh1[first_y:last_y+1, first_x:last_x+1]
    thresh2 = 255 - thresh2

    # get external contours and keep largest one
    contours = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # make mask from contour
    mask = np.zeros_like(thresh2 , dtype=np.uint8)
    drawcont = cv2.drawContours(mask, [big_contour], 0, 255, -1)

    # make crop black everywhere except where largest contour is white in mask
    result = crop.copy()# img.copy() 
    result[mask==0] = 0
    # reshape to 244,244
    if img.shape[1] != result.shape[1]:
        result = np.c_[result,np.zeros((result.shape[0],(img.shape[1] - result.shape[1])))]
    if img.shape[0] != result.shape[0]:
        result = np.append(result, np.zeros(((img.shape[0] - result.shape[0]),result.shape[1])), axis=0)

    return result


def segment_breast_16b(image, threshold = 25, refactor = 257):
    # remove totally white pixels - artifacts
    img = image.copy()
    img[img==img.max()] = 0

    breast_side = determine_side(img, threshold)
    gray = img.copy()

    # threshold and invert
    thresh1 = cv2.threshold(gray, threshold, gray.max(), cv2.THRESH_BINARY)[1]
    thresh1 = gray.max() - thresh1

    # remove borders
    # count number of white pixels in columns as new 1D array
    count_cols = np.count_nonzero(thresh1, axis=0)
    # crop image
    if breast_side == 'l':
        # get first and last x coordinate where black
        first_x = 0
        last_x = np.where(count_cols>0)[0][-1]
    else:
        # get first and last x coordinate where black
        first_x = np.where(count_cols>0)[0][0]
        last_x = img.shape[0]-1

    # count number of white pixels in rows as new 1D array
    count_rows = np.count_nonzero(thresh1, axis=1)
        # get first and last y coordinate where black
    first_y = max(np.where(count_rows>0)[0][0],5)
    last_y = min(np.where(count_rows>0)[0][-1],img.shape[1]-5)
    crop = img[first_y:last_y+1, first_x:last_x+1]

    # crop thresh1 and invert
    thresh2 = thresh1[first_y:last_y+1, first_x:last_x+1]
    thresh2 = gray.max() - thresh2

    # get external contours and keep largest one
    contours = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # make mask from contour
    mask = np.zeros_like(thresh2 , dtype=np.uint16)
    drawcont = cv2.drawContours(mask, [big_contour], 0, gray.max(), -1)

    # make crop black everywhere except where largest contour is white in mask
    result = crop.copy()# img.copy() 
    result[mask==0] = 0
    # reshape to 244,244
    if img.shape[1] != result.shape[1]:
        result = np.c_[result,np.zeros((result.shape[0],(img.shape[1] - result.shape[1])))]
    if img.shape[0] != result.shape[0]:
        result = np.append(result, np.zeros(((img.shape[0] - result.shape[0]),result.shape[1])), axis=0)

    return result