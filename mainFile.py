from functions import *
###################################################### OPTIONS ########################################################
num_bilateral = 7  # number of bilateral filtering steps
num_down = 2  # number of downsampling steps
contourThreshold = 25
additiveSaturation =0


def mainfunc(path):
    img_rgb = cv2.imread(path)
# downsample image using Gaussian pyramid
    img_color = img_rgb
    img_rgb  = cv2.medianBlur(img_rgb,7)

    for _ in range(num_down):
        img_color = pyramidDown(img_color)


    for _ in range(num_bilateral):
        img_color = bilateralfilter(img_color, cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY),
                                    sigma_s=9,
                                    sigma_r=7)

    # upsample image to original size
    for _ in range(num_down):
        img_color = pryamidUp(img_color)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                    blockSize=9,
                                     C=3)
    # img_edge = cv2.Canny(img_blur,80,255)
  #  img_edge = 255- img_edge
   # img_edge = adaptive_thresh(img_blur,9,4)
    img_blur = img_edge.copy()
    # contours extraction
    temp_img = np.ones(img_blur.shape)
    ret, thresh = cv2.threshold(img_blur, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(img_blur.astype('uint8') // 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursModified = []
    for i in contours:
        if i.shape[0] >= contourThreshold:
            contoursModified.append(i)
    contours = contoursModified
    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)
    temp_img2 = np.ones(temp_img.shape)
    cv2.fillPoly(temp_img2, pts=contours, color=(0, 0, 0))
    temp_img2 = 1 - temp_img2
    temp_img2 = cv2.dilate(temp_img2,np.ones((2,2),np.uint8))
    temp_img = temp_img2
    temp_img = temp_img.astype('uint8') * 255
    img_color = cv2.medianBlur(img_color, 7)
    img_color = np.array(img_color)
    img_blur = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB)
    x = min(img_color.shape[0], img_blur.shape[0])
    y = min(img_color.shape[1], img_blur.shape[1])
    img_cartoon = np.bitwise_and(img_color[:x, :y, :], img_blur[:x,:y,:])
    img_cartoon = increase_brightness(img_cartoon, 40)
    img_cartoon = img_cartoon // 24 * 24
    img_cartoon = cv2.cvtColor(img_cartoon, cv2.COLOR_RGB2HSV)
    satChannel = img_cartoon[:, :, 1].copy()
    satChannel[satChannel < 180] += additiveSaturation
    #satChannel[satChannel > 30] -= additiveSaturation
    img_cartoon[:, :, 1] = satChannel
    img_cartoon = cv2.cvtColor(img_cartoon, cv2.COLOR_HSV2RGB)
    cv2.imwrite('results/'+ path.split('/')[-1], img_cartoon)
    showImage(img_rgb)
    cv2.waitKey()
    showImage(img_cartoon)
    cv2.waitKey()
    cv2.destroyAllWindows()