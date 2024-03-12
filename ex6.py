import cv2
import numpy as np
import matplotlib.pyplot as plt


#part 6.1
def gaussian_1D_Kernel(sigma):
    '''
    takes sigma = gaussian width
    returns
    g: the 1D gaussian kernel
    gd: the derivative of g

    normalize g such that it sums to 1.
    '''
    #using 5sigma rule
    length = int(2*5*sigma + 1)
    if length % 2 == 0:
        length += 1  #to make sure legnth is odd
    
    x = np.arange(length) - length//2  #these are the kernel indices

    #gaussian function 
    g = np.exp(-(x**2)/ (2*sigma**2))
    g /= g.sum() #normalized to sum to 1
    #derivative
    gd = -x / (sigma**2) *g

    return g, gd

#part 6.2
def gaussian_smoothing(im, sigma):
    '''
    returns I: gaussian smoothed image of im
    Ix and Iy: gaussian smoothed derivatioves of im
    '''
    g, gd = gaussian_1D_Kernel(sigma)
    I = cv2.sepFilter2D(im, -1, g, g)
    Ix = cv2.sepFilter2D(im, -1, gd, g)
    Iy = cv2.sepFilter2D(im, -1, g, gd)
    return I, Ix, Iy

#part 6.3
def structure_tensor(im, sigma, epsilon):
    '''
    returns matrix C(x,y)
    '''
    g_epsilon, _ = gaussian_1D_Kernel(epsilon)
    I, Ix, Iy    = gaussian_smoothing(im, sigma)
    Ixx = cv2.sepFilter2D(Ix**2, -1, g_epsilon, g_epsilon)
    Ixy = cv2.sepFilter2D(Ix*Iy, -1, g_epsilon, g_epsilon)
    #Iyx = cv2.sepFilter2D(Ix**2, -1, g_epsilon, g_epsilon)
    Iyy = cv2.sepFilter2D(Iy**2, -1, g_epsilon, g_epsilon)

    C = np.array([[Ixx, Ixy],
                  [Ixy, Iyy]]
                )
    return C


#party 6.4
def harris_measure(im, sigma, epsilon, k):
    '''computer the harris corner measure for each pixel in the image'''
    '''r is the harris corner for each pixel'''
    C = structure_tensor(im, sigma, epsilon)
    a = C[0][0]
    b = C[1][1]
    c = C[0][1]
    r = a*b - c**2 - k*((a+b)**2)
    return r

#part 6.5
def corner_detector(im, sigma, epsilon, k, tau):
    ''' 
    Detects corners in an image using the Harris corner measure and non-maximum suppression.
    c: List of (x, y) coordinates of detected corners.
    '''
    r = harris_measure(im, sigma, epsilon, k)
    # Initialize the suppressed response matrix with zeros
    suppressed = np.zeros_like(r)
    
    # Iterate over each pixel excluding the border
    for y in range(1, r.shape[0] - 1):
        for x in range(1, r.shape[1] - 1):
            # Get the local patch
            patch = r[y-1:y+2, x-1:x+2]
            # Check if the current pixel is the maximum in its neighborhood
            if r[y, x] == np.max(patch) and r[y, x] > tau:
                suppressed[y, x] = r[y, x]
    corners = np.where(suppressed > tau)
    c = list(zip(corners[1], corners[0]))
    return c

#MAIN

im = cv2.imread("week06_data/TestIm2.png", cv2.IMREAD_GRAYSCALE).astype(float)
#g, gd = gaussian_1D_Kernel(2)
#print(f"{g} \n {gd}")
plt.imshow(im, cmap='gray')
plt.show()
sigma   = 2 # if sigma = 0 then all blurred images are the same. Cuz we are just sampling the value
epsilon = 2
k, tau  = 0.04, 0.1
im_blur, x_blur, y_blur = gaussian_smoothing(im, sigma)
plt.imshow(im_blur, cmap='gray')
plt.show()
plt.imshow(x_blur)
plt.show()
plt.imshow(y_blur)
plt.show()

C = structure_tensor(im, sigma,epsilon)
print(C)

# Normalize for visualization

# Display the results
plt.imshow(C[0,0])
plt.show()
plt.imshow(C[0,1])
plt.show()
plt.imshow(C[1,1])
plt.show()

r = harris_measure(im, sigma, epsilon, k)
print(f"r is {r}\n")
plt.imshow(r)
plt.show()

corners = corner_detector(im, sigma, epsilon, k, tau)
for x, y in corners:
    cv2.circle(im, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
print("part 6.5")
plt.imshow(im)
plt.show()


#part 6.6
# Setting parameter values 
t_lower = 50  # Lower Threshold 
t_upper = 150  # Upper threshold 
im2 = cv2.imread("week06_data/TestIm2.png", cv2.IMREAD_GRAYSCALE)
im  = cv2.imread("week06_data/TestIm1.png", cv2.IMREAD_GRAYSCALE)
# Applying the Canny Edge filter 
edge  = cv2.Canny(im , t_lower, t_upper)
edge2 = cv2.Canny(im2, t_lower, t_upper) 
edge3 = cv2.Canny(im2, 10, t_upper)
edge4 = cv2.Canny(im2, t_lower, 100)
edge5 = cv2.Canny(im2, 150, 300)


plt.imshow(edge2)
plt.show()
plt.imshow(edge3)
plt.show()
plt.imshow(edge4)
plt.show()
plt.imshow(edge5)
plt.show()