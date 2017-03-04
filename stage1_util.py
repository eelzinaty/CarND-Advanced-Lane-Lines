import numpy as np
import cv2
import glob
import pickle

def calibrate(cal_path, size=(9,6)):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(cal_path)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, size, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, size, corners, ret)
            write_name = 'output_images/camera_cal/corners_found_'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)

    # Test undistortion on an image
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/camera_cal/test_undist.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "output_images/camera_cal/wide_dist_pickle.p", "wb" ) )
    
    return mtx, dist

def load_camera_cal():
    dist_pickle = pickle.load( open( "output_images/camera_cal/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#handcrafted road points for perspective projection matrix calculation
src = np.float32 ([
        [220, 651],
        [350, 577],
        [828, 577],
        [921, 651]
    ])

dst = np.float32 ([
        [220, 651],
        [220, 577],
        [921, 577],
        [921, 651]
    ])

#h = 1280 #exampleImg_undistort.shape[:2]
#w = 720

# define source and destination points for transform
#src = np.float32([(575,464),
#                  (707,464), 
#                  (258,682), 
#                  (1049,682)])
#dst = np.float32([(450,0),
#                  (w-450,0),
#                  (450,h),
#                  (w-450,h)])

# calculating front view -> top view projection matrix
M_perspective_720 = cv2.getPerspectiveTransform (src, dst)
# calculating top -> front view matrix
M_perspective_inv_720 = cv2.getPerspectiveTransform (dst, src)

def corners_unwarp(img, nx, ny, mtx, dist):
    img_size = (img.shape [1], img.shape [0])
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M_perspective_720, img_size, flags=cv2.INTER_LINEAR)
     
    return warped, M_perspective_inv_720

def morphology_filter (image, s_thresh=(-.1, -.035)):
    """Filtering lane lines with "opening" morphology operation
    
        1. Taking linear combination of
            HLS S layer * 0.6 + Grayscaled image * 0.4
    
        2. Applying opening, as a result we have image with erased lane lines 
        
        3. Subtracting image without lane lines from original image and
            returning lane lines
    
    Args:
        image (np.array): color image in BGR format
    Returns:
        np.array: graysacle image with filtered lane lines along with some another parts of the image
    """
    
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    hls_s = hls[:,:,2]
    #hls_s = get_s_from_hls (image)
    src = hls_s * 0.6 + gray * 0.4

    src = np.array( src / 255.).astype ('float32') - 0.5

    blurf = np.zeros((1, 5))
    blurf.fill (1)
    src = cv2.filter2D(src, cv2.CV_32F, blurf)

    f = np.zeros((1, 30))
    f.fill (1)
    l = cv2.morphologyEx(src, cv2.MORPH_OPEN, f)

    filtered = src - l
    #kernel = np.ones((5,5),np.float32)/25
    #dst = cv2.filter2D(filtered,-1,kernel)
    filtered = cv2.medianBlur(filtered,5)
    #print(filtered)
    # Threshold color channel
    f_binary = np.zeros_like(filtered)
    f_binary[(filtered >= s_thresh[0]) & (filtered <= s_thresh[1])] = 1
    
    return f_binary

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255), space = 'gray', ch=0):
    # Convert to grayscale
    gray = img
    if space == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif space == 'hls':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    elif space == 'hsv':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    elif space == 'yuv':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), space = 'gray', ch=1):
    # Convert to grayscale
    gray = img
    if space == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif space == 'hls':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    elif space == 'hsv':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hsv[:,:,ch]
    elif space == 'yuv':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2), space = 'gray', ch=1):
    # Convert to grayscale
    gray = img
    if space == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif space == 'hls':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    elif space == 'hsv':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    elif space == 'yuv':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #H = hls[:,:,0]
        #L = hls[:,:,1]
        gray = hls[:,:,ch]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(140, 255), sx_thresh=(30, 100)):
    img = np.copy(img)
    # Choose a Sobel kernel size
    ksize = 15 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=sx_thresh, space = 'yuv', ch=0)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=sx_thresh, space = 'yuv', ch=0)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=sx_thresh, space = 'yuv', ch=0)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.7, 1.3), space = 'yuv', ch=0)

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((gradx == 1) & (grady == 1) & (dir_binary == 1))] = 1
    #scaled_combined = np.uint8(255*combined/np.max(combined))
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float)
    #l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,1]
    # Sobel x
    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    #abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    #scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    #sxbinary = np.zeros_like(scaled_sobel)
    #sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(combined), combined, s_binary))
    
    d_binary = np.zeros_like(combined)
    d_binary[(s_binary == 1) | (combined == 1)] = 1
    
    return d_binary, color_binary
    







