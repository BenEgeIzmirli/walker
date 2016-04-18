
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import sys
import pickle

# This contains a number of functions and classes I've written
# that are relevant to evaluating dynamic and secondary spectra.
import pulsar as psr


# This code takes an image on the sky, and generates a secondary spectrum from it, using the geometry derived in Walker et al 2006. To generate the image on the sky, two steps are taken - first an image with random "speckles" is created, then each speckle is itself treated as an image and made into further speckles. The anisotropy, angle, spatial distribution, and intensity distribution can be specified at each speckling stage.
# 
# This code makes use of "Indexed2D" objects, which can be found in my "pulsar.py" dynamic/secondary spectrum package. The point of these objects is that they are accessed using the axis values instead of integer indices. Here is some documentation:
## making the object:

myarr = [[0,1,2],
         [3,4,5],
         [6,7,8]]  # a 3x3 array

y_axis = [-1,0,1]  # the y axis
x_axis = [0,1,2]   # the x axis

test = psr.Indexed2D(data=myarr,axes=(y_axis,x_axis)) # make the Indexed2D object

# or, equivalently:

test = psr.Indexed2D()
test.set_data(myarr)
test.set_axes((y_axis,x_axis))

## Accessing the points:

test[-1,1]  # gives 1
test[1,2]   # gives 8
test[0,0]   # gives 3

test.get_data()   # returns the data as a numpy array
test.get_axes()   # returns both axes as a tuple
test.get_x_axis() # returns the x axis
test.get_y_axis() # returns the y axis

## Setting points:

test.set_point(10,(-1,2))
# test is now:
# [[0,1,10],
#  [3,4,5],
#  [6,7,8]]

## plotting:

test.show() # does a matplotlib imshow command on the data and axes
plt.show()
# In[ ]:

# This is a worker function to be passed to the
# multiprocessing kernels
def crunchy(index,image):
    #sys.stdout.write(str(index))
    y_axis = image.get_y_axis()
    x_axis = image.get_x_axis()
    max_q = max(x_axis)*2
    max_p = 2*np.sqrt(max(y_axis)**2+max(x_axis)**2)
    
    sec_y = np.linspace(max_p,-max_p,2*len(y_axis)-1)
    sec_x = np.linspace(-max_q,max_q,2*len(x_axis)-1)
    data = np.zeros((len(sec_y),len(sec_x)))
    
    ret = psr.Indexed2D(data=data,axes=(sec_y,sec_x))
    
    for y in y_axis:
        for x in x_axis:
            if image[y,x]==0:
                continue
            q = float(x - index[1])
            #print(index[1],x,q)
            p = float((y**2 + x**2)-(index[0]**2 + index[1]**2))
            value = image[y,x]*image[index[0],index[1]]
            #print(y,x,p,q,value)
            ret.set_point(value,(p,q))
    return ret

# This is the main function that will evaluate the secondary spectrum
# given an image.
# 
# Parameters:
#    -image : an Indexed2D object containing the image on the sky
#    -num_threads : the number of parallel processing threads to use
# Returns:
#    -an Indexed2D object containing the secondary spectrum
def calc_sec(image,num_threads=mp.cpu_count()-1):
    indices = []
    y_axis = image.get_y_axis()
    x_axis = image.get_x_axis()
    
    for y in y_axis:
        for x in x_axis:
            if image[y,x]==0:
                continue
            indices.append((y,x))
    
    print("calculating secondary spectrum with",len(indices),"points...")
    print("using",num_threads,"processes...")
    
    pool = mp.Pool(processes=num_threads)
    output = pool.map(partial(crunchy, image=image), indices)
    
    sec = np.zeros(np.shape(output[0].get_data()))
    sec_y = output[0].get_y_axis()
    sec_x = output[0].get_x_axis()
    for o in output:
        sec = np.add(sec,o.get_data())
    
    sec = psr.Indexed2D(data=sec,axes=(sec_y,sec_x))
    sec.set_point(0,(0,0))
    print("Done")
    return sec

# Gives coordinates and strength for a single point from the specified
# tilted random anisotropic distribution.
# 
# Parameters:
#    -anisotropy : the ratio of the minor and major axes - higher is more anisotropic
#    -angle : the angle of the anisotropy
#    -random_func : a function that will give a random point from some probability
#        distribution
#    -strength_func : a function that will map coordinates to strength - points farther
#        from the origin will be weaker, for instance.
# Returns:
#    -the coordinates and strength of the random point
def tilted_random(anisotropy,angle,
                  random_func=np.random.standard_cauchy,
                  strength_func=lambda y,x: psr.gaussian(np.sqrt(x**2+y**2),0,1)):
    
    # Choose the point from an anisotropic distribution with angle 0
    y = random_func()*(1/float(anisotropy))
    x = random_func()
    
    # Rotate the point by angle
    total = np.sqrt(y**2 + x**2)
    this_angle = np.arctan(y/x)
    y = (y/np.sin(this_angle))*np.sin(this_angle + angle*np.pi/180.)
    x = (x/np.cos(this_angle))*np.cos(this_angle + angle*np.pi/180.)
    
    strength = strength_func(y,x)
    return y,x,strength

# This function simulates simple scintillation in two stages.
# The first stage makes some number of speckles about the direct
# line of sight, with some anisotropy, and the second stage
# makes some number of speckles about each first-stage speckle,
# with some anisotropy.
#
# Parameters:
#    -num_speckles : the number of speckles that will be created in that stage.
#    -anisotropy : the ratio of the minor and major axes - higher is more anisotropic
#    -angle : the angle of the anisotropy
#    -random_func : a function that will give a random point from some probability
#        distribution. It will determine the 
#    -strength_func : a function that will map coordinates to strength - points farther
#        from the origin will be weaker, for instance.
#    -image_size_px : the sky image is by default a 128x128 pixel image.
#    -image_range : by default the image is in units of scattering disk - a separation
#        of one unit in the image is one scattering disk width. By default image_range=3,
#        meaning that points that lie more than 3 scattering disk widths away in either
#        x or y will not be in the image.
# Returns:
#    -an Indexed2D object containing the simulated image on the sky
def scintillation(num_speckles_1, anisotropy_1, angle_1,
                  num_speckles_2, anisotropy_2, angle_2,
                  random_func_1=np.random.standard_cauchy,
                  random_func_2=np.random.standard_cauchy,
                  intensity_func_1=lambda y,x: psr.gaussian(np.sqrt(x**2+y**2),0,1),
                  intensity_func_2=lambda y,x: psr.gaussian(np.sqrt(x**2+y**2),0,1),
                  image_size_px=128, image_range=3):
    
    # construct the return array to be populated
    data = np.zeros((image_size_px-1,image_size_px-1))
    x_axis = np.linspace(-image_range,image_range,image_size_px-1)
    y_axis = np.linspace(image_range,-image_range,image_size_px-1)
    ret = psr.Indexed2D(data=data,axes=(y_axis,x_axis))
    
    for i in range(num_speckles_1):
        # get the coordinates and intensity of primary speckle
        y_1, x_1, i_1 = tilted_random(anisotropy_1, angle_1, random_func_1, intensity_func_1)
        
        for j in range(num_speckles_2):
            # get the coordinates and intensity of secondary speckle
            y_2, x_2, i_2 = tilted_random(anisotropy_2, angle_2, random_func_2, intensity_func_2)
            
            # final coordinates on the return image
            y = y_2 + y_1
            x = x_2 + x_1
            
            # Don't try to write any points outside of the image
            if np.absolute(y)>image_range or np.absolute(x)>image_range:
                continue
            
            # The intensity is set as the product of the intensities of
            # the primary and secondary speckles.
            intensity = i_1*i_2 + ret[y,x]
            ret.set_point(intensity, (y,x))
    return ret


# In[ ]:

num_1 = 10
anisotropy_1 = 100
angle_1 = -60
num_2 = 50
anisotropy_2 = 100
angle_2 = 30

scint_image = scintillation(num_1, anisotropy_1, angle_1, num_2, anisotropy_2, angle_2)

scint_image.show()
plt.show()


# In[ ]:

sec = calc_sec(scint_image,num_threads=15)
sec_copy = sec
sec_copy2 = sec


# In[ ]:

scint_image.show()
#sec.set_point(10,(0.5,-0.03))
sec.show()
plt.show()


# In[ ]:

pickle.dump( (scint_image,sec), open( "walker.p", "wb" ) )


# In[ ]:



