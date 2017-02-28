from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import cv2

#Thank you, Alan Bernstein
#https://gist.github.com/alanbernstein/d34ced4786d24bdc20bc6b923ba33308
    
def quadratic_ransac_fit (X, y, residual_threshold = 60):
    """Function performs polynomial RANSAC fitting
    Args:
        X (list): list of X values
        y (list): list of corresponding y values
        residual_threshold (float): Maximum residual for a data sample to be classified as an inlier.
                                    By default the threshold is chosen as the MAD (median absolute deviation)
                                    of the target values y.
                                    see http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    Returns:
        polynomial coeffs (list), inlier_mask (list), outlier_mask (list)
        polynomial coeffs - coefficients a, b, c for a + b*x + c*x**2 = y
        inlier_mask - inlier mask for input lists
        outlier_mask - outlier mask
    """

    x_ = X.reshape ((-1 ,1))
    y_ = y.reshape ((-1, 1))

    xi = np.linspace (min(x_), max(x_), 100).reshape ((-1, 1))

    poly = PolynomialFeatures (degree=2)
    x_2 = poly.fit_transform (x_)
    xi_2 = poly.fit_transform (xi)

    model = linear_model.RANSACRegressor (linear_model.LinearRegression (), residual_threshold = residual_threshold)
    try:
        model.fit (x_2, y_)
    # ValueError: No inliers found, possible cause is setting residual_threshold (None) too low
    except ValueError:
        return None, None, None
    
    yi = model.predict (xi_2)
    c = model.estimator_.coef_
    yi_b = np.dot (c, xi_2.T).T

    c_b = np.array ([float(yi[3][0] - yi_b[0]), c[0, 1], c[0, 2]])
    yi_b = np.dot (c_b, xi_2.T).T

    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not (inlier_mask)
    
    return c_b, inlier_mask, outlier_mask

def plyfit_lane_line_ransac (lane_mask):
    """Polynomial fitting with RANSAC
    Args:
        lane_mask (np.array): grayscale thresholded image
    Returns:
        found(boolean), coeffs(list)
    """
    x, y = np.where (lane_mask > 0)
    if len (x)>3:
        coeffs, inlier_mask, outlier_mask = quadratic_ransac_fit (x, y)
        if (coeffs is None):
            return False, [0, 0, 0]
        return True, coeffs.tolist ()
    else:
        return False, [0, 0, 0]

def filter_by_hist_maximum (lanes_mask_in, interval_size):
    """Filter thresholded image by histogram maximum value. Histogram calculates
        as count of white pixels with fixed x
    Args:
        lanes_mask_in (no.array): grayscale thresholded image
        interval_size (int): all pixels not in interval (max x - interval_size, max x + interval_size) will be set 0
    Returns:
        filtered grayscale thresholded image
    """
    print(lanes_mask_in)
    lanes_mask = np.copy (lanes_mask_in)
    filtered_lanes_mask = np.zeros_like (lanes_mask)
    
    histogram = np.sum(lanes_mask[int(lanes_mask.shape[0]/2):,:], axis=0)
    
    max1 = np.argmax (histogram)
    w  = np.where (lanes_mask > 0)
    
    (x, y) = [w]
        
    ids = np.where (
        (y < max1 + interval_size) &
        (y > max1 - interval_size)
    )
    
    filtered_lanes_mask [(x[ids], y[ids])] = lanes_mask_in [(x[ids], y[ids])]
    
    return filtered_lanes_mask

def filter_two_maximums (lanes_mask, interval_size):
    """Filter thresholded image by 2 highest histogram maximum value. Histogram calculates
        as count of white pixels with fixed x
    Args:
        lanes_mask_in (no.array): grayscale thresholded image
        interval_size (int): all pixels not in interval (max x - interval_size, max x + interval_size) will be set 0
    Returns:
        (image1, image2)
        two images with filtered two maximums
    """

    max1 = filter_by_hist_maximum (lanes_mask, interval_size)
    all_without_first_maximum = np.copy (lanes_mask)
    all_without_first_maximum [np.where(max1 > 0)] = 0

    max2 = filter_by_hist_maximum (all_without_first_maximum, interval_size)

    return max1, max2

def calc_lanes_coeffs_ransac (lanes_mask):
    """Fits lane lines with RANSAC
    Args:
        lanes_mask (no.array): grayscale thresholded image
    Returns:
        coeffs1(list), coeffs2(list), two_maximums_filtered_mask(image)
    """

    max1, max2 = filter_two_maximums (lanes_mask, 150)
    maxs_mask = np.add (max1, max2)

    found1, coeff1 = plyfit_lane_line_ransac (max1)
    found2, coeff2 = plyfit_lane_line_ransac (max2)
    
    if (not found1):
        coeff1 = None
    
    if (not found2):
        coeff2 = None
    
    if (found1 and found2):
        if (abs(coeff1[0]) > abs(coeff2[0])):
            return coeff2, coeff1, maxs_mask

    return coeff1, coeff2, maxs_mask

def filter_by_polyline (lanes_mask_in, coeff, interval_size):
    """Filter thresholded image by specified polynomial
    Args:
        lanes_mask_in (no.array): grayscale thresholded image
        coeff(list): (a, b, c) - coefficients of polynomial
        interval_size (int): all pixels not in interval (max x - interval_size, max x + interval_size) will be set 0
    Returns:
        filtered_mask(image)
    """
    lanes_mask = np.copy (lanes_mask_in)
    filtered_lanes_mask = np.zeros_like (lanes_mask)
    
    x, y = np.where (lanes_mask > 0)
    
    a, b, c = coeff
    
    ids = np.where (
        (y < a + b*x + c*x*x + interval_size) &
        (y > a + b*x + c*x*x - interval_size)
    )
    
    filtered_lanes_mask [(x[ids], y[ids])] = lanes_mask_in [(x[ids], y[ids])]
    
    return filtered_lanes_mask

def plyfit_lane_mse (lane_mask):
    """Polynomial fitting with means squared error method
    Args:
        lane_mask (np.array): grayscale thresholded image
    Returns:
        tuple: in form of [found, a, b, c]
    """
    x, y = np.where (lane_mask > 0)
    if len (x)>3:
        coeffs = np.polyfit(x, y, 2)
        return [True] + coeffs.tolist ()
    else:
        return [False, 0, 0, 0]

def calc_lanes_coeffs_mse (lanes_mask):
    """Fits lane lines with mean squared error method
    Args:
        lanes_mask (no.array): grayscale thresholded image
    Returns:
        coeffs1(list), coeffs2(list), plynmial_masked_image(image)
    """

    # begin with ransac
    coeff1_ransac, coeff2_ransac, maxs_mask = calc_lanes_coeffs_ransac (lanes_mask)
    if (coeff1_ransac != None):
        poly_mask_1 = filter_by_polyline (lanes_mask, coeff1_ransac, 80)
    else:
        poly_mask_1 = np.zeros_like (lanes_mask)
    
    # filtering with found ransac polynomial
    if (coeff2_ransac != None):
        poly_mask_2 = filter_by_polyline (lanes_mask, coeff2_ransac, 80)
    else:
        poly_mask_2 = np.zeros_like (lanes_mask)
        
    poly_mask = np.add (poly_mask_1, poly_mask_2)
    
    # fitting with mse
    if (coeff1_ransac == None):
        coeff1 = None
    else:
        found, c, b, a = plyfit_lane_mse (poly_mask_1)
        if (not found):
            coeff1 = None
        else:
            coeff1 = [a, b, c]
    
    if (coeff1_ransac == None):
        coeff2 = None
    else:
        found, c, b, a = plyfit_lane_mse (poly_mask_2)
        if (not found):
            coeff2 = None
        else:
            coeff2 = [a, b, c]
    
    # choosing left and right lane line where possible
    if (
        coeff1 != None and
        coeff2 != None
    ):
        if (abs(coeff1[0]) > abs(coeff2[0])):
            return coeff2, coeff1, poly_mask

    return coeff1, coeff2, poly_mask


def getXYQuadratic (coeffs, limits):
    """Calcs polynomial points from cefficients
    """
    x = np.linspace (limits[0], limits[1], 100)
    y = np.add(np.add(coeffs [0], coeffs [1] * x), coeffs [2] * x * x)
    return x, y

def get_lane_lines_field (img, coeff1, coeff2):
    """Draws lane and lane lines by coeffs
    """
    zero_layer = np.zeros_like (img).astype (np.uint8)
    
    if (coeff1 == None or coeff2 == None):
        return np.dstack ((zero_layer, zero_layer, zero_layer))

    # from tips and tricks from lesson
    left_x, left_y = getXYQuadratic (coeff1, (0, 1280))
    right_x, right_y = getXYQuadratic (coeff2, (0, 1280))
    pts_left = np.array ([np.transpose(np.vstack([left_y, left_x]))])
    pts_right = np.array ([np.flipud(np.transpose(np.vstack([right_y, right_x])))])
    pts = np.hstack ((pts_left, pts_right))

    #lanes_drown = np.dstack ((zero_layer, zero_layer, zero_layer))
    lanes_drown = cv2.fillPoly (img, np.int_([pts]), (0, 255, 0))
    
    lanes_drown = draw_lane_line_top_view (lanes_drown, coeff1)
    lanes_drown = draw_lane_line_top_view (lanes_drown, coeff2)
    
    return lanes_drown

def draw_lane_line_top_view (img, coeffs):
    """Draw one lane line
    """
    x, y = getXYQuadratic (coeffs, (0, 1280))
    pts = np.array ([np.transpose(np.vstack([y, x]))]).astype(np.int32)
    return cv2.polylines(img, [pts], False, (255, 0, 0), 20)
