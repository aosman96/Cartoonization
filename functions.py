import cv2
import numpy as np
from sklearn import cluster
# sigma_s : sigma used in spatial gaussian
# sigma_r : sigma used in intensity gaussian
######################################################################################
#######################################################################################

def showImage(img,title='',resize=True,scale=1/4,multiplyBy = 1):
    imgNew = img * multiplyBy
    if resize:
        imS = cv2.resize(imgNew, (int(imgNew.shape[0] * scale), int(imgNew.shape[1] * scale)))  # Resize image
        cv2.imshow(title, imS)  # Show image
    else:
        cv2.imshow(title, imgNew)  # Show image

def bilateralfilter(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
    # Symmetric padding : pads along the reflected mirror of edge of the array
    # Ex: a [1,2,3,4,5]
    # np.pad(a, (2,3), 'symmetric') means pad 2 elements of first axis edge and 3 elements of second axis edge
    # result: [2,1,1,2,3,4,5,5,4,3]
    # Pads first&second edges of each dimension with 3*sigma (r) for applying filter on borders
    if image.ndim == 3:
        h, w, ch = image.shape
        I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif image.ndim == 2:
        h, w = image.shape
        I = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
    else:
        print('Input image is not valid!')
        return image
    # Check texture size equals given image size then do padding
    if texture.ndim == 3:
        ht, wt, cht = texture.shape
        # If texture shape is not equal to image shape, return
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        # else pad texture
        T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif texture.ndim == 2:
        ht, wt = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    # Pre-compute
    # Create np array of zeros with the same shape of the image
    output = np.zeros_like(image)
    # e^(- x / 2sigma^2)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel (COLOR)
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
    # Generate a spatial Gaussian function (cutoff 6-sigma)
    # -r for symmetric grid Ex: 0->6 becomes -3 -> 3
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    # Create multi-variate gaussian distribution for spatial domain with x,y
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    # Main body
    if I.ndim == 2 and T.ndim == 2:  # I1T1 (2D Image, 2D Texture) filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                # Get gaussian values representing weights for the window
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                # Calculate the intensity of the current pixel using the weighted gaussian values
                # for j=-3sigma->3sigma sum(w(j) * I(j))/sum(w)
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif I.ndim == 3 and T.ndim == 2:  # I3T1 (3D Image, 2D Texture) filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 3 and T.ndim == 3:  # I3T3 (3D Image, 3D Texture) filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                # Product of 3 independent gaussians for each channel RGB
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 2 and T.ndim == 3:  # I1T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return image

    # return np.clip(output, 0, 255)
    return output


# Downsamples image to smaller size (1/4 of size each function call)
def pyramidDown(img):
    gaussianKernel = [[1, 4, 6, 4, 1],
                      [4, 16, 24, 16, 4],
                      [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4],
                      [1, 4, 6, 4, 1]]
    gaussianKernel = np.array(gaussianKernel)
    gaussianKernel = (1 / 256) * gaussianKernel
    img = cv2.filter2D(img, -1, gaussianKernel)
    img = img[1:img.shape[0]:2, 1:img.shape[1]:2, :]
    return img


# Upsamples given image to twice its original size and approximates missing pixels
def pryamidUp(img):
    gaussianKernel = [[1, 4, 6, 4, 1],
                      [4, 16, 24, 16, 4],
                      [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4],
                      [1, 4, 6, 4, 1]]
    gaussianKernel = np.array(gaussianKernel)
    gaussianKernel = (4 / 256) * gaussianKernel
    rows = img.shape[0]
    cols = img.shape[1]
    rows *= 2
    cols *= 2
    newImg = np.ones((rows, cols, img.shape[2]))
    newImg[0:newImg.shape[0]:2, 0:newImg.shape[1]:2, :] = 0
    newImg[1:newImg.shape[0]:2, 1:newImg.shape[1]:2, :] = img
    newImg[newImg > 255] = 255
    newImg[newImg < 0] = 0
    newImg = cv2.filter2D(newImg, -1, gaussianKernel)
    newImg = newImg.astype("uint8")
    return newImg

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def clusterColours(img, k=4):
    [x, y, z] = img.shape
    img2d = img.reshape(x * y, z)

    ## TODO: Construct kmeans object using cluster.KMeans with appropriate n_clusters
    kMeansObject = cluster.KMeans(k)
    print("Finished Clustering with K =", k)

    ## TODO: Fit the kmeans object with the data of the image
    kMeansObject.fit(img2d)

    labels = kMeansObject.labels_
    centers = kMeansObject.cluster_centers_

    indexArray = labels.reshape(x, y)
    for i in range(k):
        img[indexArray == i] = centers[i]

    return img

#adaptive thresholding using integral images
def adaptive_thresh(input_img, windowsize, C):

    h, w = input_img.shape
    S = windowsize
    s2 = S//2
    #integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
  #  for col in range(w):
    #    for row in range(h):
    #        int_img[row,col] = input_img[0:row,0:col].sum()

    int_img = input_img.copy()
    for i in range(int_img.ndim):
        int_img=int_img.cumsum(axis=i)

    #output img
    out_img = np.zeros_like(input_img)

    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = max(row-s2, 0)
            y1 = min(row+s2, h-1)
            x0 = max(col-s2, 0)
            x1 = min(col+s2, w-1)

            count = (y1-y0)*(x1-x0)

            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]

            if input_img[row, col]*count < sum_*(100. - C) / 100:
                out_img[row,col] = 0
            else:
                out_img[row,col] = 255

    return out_img