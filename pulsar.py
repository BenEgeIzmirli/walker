import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from astropy.io import fits
import numpy as np
import os
import sys
import multiprocessing as mp
from multiprocessing_helper_functions import *

# Returns a sorted list of all files in the given directory.
# Takes:
#    a directory name
# Returns:
#    all files in that directory
def get_file_list(d):
    files = []
    
    os.chdir(d)
    for f in os.listdir('.'):
        if os.path.isfile(f):
            print('    -using: '+f)
            files.append(d+f)
                
    return sorted(files)

# Shows an image with the given X and Y axes.
# Takes:
#    showme - the 2D array to be shown.
#    axis_y - a 1D array containing the y axis to be used in the plot.
#    axis_x - a 1D array containing the x axis to be used in the plot.
# Returns:
#    Plots the image, returns nothing. Note the show() call must be made manually.
def show_image(showme, axis_y=None, axis_x=None):
    try:
        if axis_y == None:
            axis_y = showme.y_axis
        if axis_x == None:
            axis_x = showme.x_axis
        showme = showme.data
    except Exception:
        if axis_x == None:
            axis_x = [i for i in range(len(showme[0]))]
        if axis_y == None:
            axis_y = [i for i in range(len(showme))]
    (x_min,x_max) = (min(axis_x),max(axis_x))
    (y_min,y_max) = (min(axis_y),max(axis_y))
    plt.figure()
    plt.imshow(showme, aspect='auto',extent=[x_min,x_max,y_min,y_max])
    plt.colorbar()
    return

# plots the given secondary spectrum overplotted with parabolas
# Takes:
#    sec - the secondary spectrum object to be plotted
#    a - a list of curvature (eta) values to be used for the parabolas plotted.
#        the parabolas will all have vertices at the origin.
# Returns:
#    Plots the secondary spectrum with overplotted parabolas, returns nothing.
#    Note that the show() call must be made manually.
def overplot_parabola(sec, a, hand=None):
    
    axis_x = sec.get_x_axis()
    axis_y = sec.get_y_axis()
    left_parab = []
    right_parab = []
    
    if type(a) is int or type(a) is float:
        a = [a]
    for this_a in a:
        if hand=='left' or hand is None:
            left_x = axis_x[:int(len(axis_x)/2)]
            left_y = []
            for x in left_x:
                y = this_a*x**2
                if y>max(axis_y):
                    left_y.append(None)
                else:
                    left_y.append(y)
            left_parab.append(left_y)
        if hand=='right' or hand is None:
            right_x = axis_x[int(len(axis_x)/2):]
            right_y = []
            for x in right_x:
                y = this_a*x**2
                if y>max(axis_y):
                    right_y.append(None)
                else:
                    right_y.append(y)
            right_parab.append(right_y)
    
    for i in range(len(a)):
        plt.plot(axis_x[:int(len(axis_x)/2)], left_parab[i], 'b-')
        plt.plot(axis_x[int(len(axis_x)/2):], right_parab[i], 'b-')

# returns a numpy array containing the dynamic spectrum.
# Takes:
#    filename - the filename of the dynamic spectrum
#    normalize_frequency - if set to True, then will normalize the dynamic spectrum
#        to compensate for uneven power in different frequency bins (horizontally stripey)
#    normalize_time - if set to True, then will normalize the dynamic spectrum
#        to compensate for uneven power in different time bins (vertically stripey)
# Returns:
#    the dynamic spectrum as a numpy array
def get_dynamic_spectrum(filename, normalize_frequency=False, normalize_time=True, outliers_sigma=5,rotate=False):
    dynamic = fits.open(filename)[0].data
    if rotate:
        dynamic = np.rot90(dynamic)
    if normalize_frequency:
        dynamic = arr_normalize_axis(dynamic,'y')
    if normalize_time:
        dynamic = arr_normalize_axis(dynamic,'x')
    if normalize_frequency or normalize_time:
        dyn_mean = np.mean(dynamic)
        dyn_std = np.std(dynamic)
        for row in range(len(dynamic)):
            for col in range(len(dynamic[row])):
                if np.abs(dynamic[row][col]-dyn_mean) > float(outliers_sigma)*dyn_std:
                    dynamic[row][col] = dyn_mean
    return dynamic

# normalizes array with respect to the given axis, making each row/column
# have the same mean within a specified region
# Takes:
#    arr - the array to be normalized
#    axis - 'y' or 'x', will specify which axis to take as the chunks
#        to be normalized
#    mask - indicates which components of the given axis are priority.
#        if mask is left None or set to an array of all the same value, then all
#        points will be considered equal priority. If mask is set to
#        ones for the first quarter, then zeroes for the rest, then
#        the array will be normalized such that the means of the rows/columns
#        will be equal within the region specified by the array. The mask can
#        also be other numbers; you could specify a list
#        [1/float(i)**2 for i in range(arr_len)], for instance.
def arr_normalize_axis(arr,axis=None,mask=None):
    if axis is None:
        return arr
    elif axis is 'y':
        if mask is None:
            mask = [1. for i in range(len(arr[0]))]
        else:
            for e in mask:
                e = np.float(e)
        mean = 0.
        for row in arr:
            mean += np.mean(mask*row)
        mean /= float(len(arr))
        for row in arr:
            this_mean = np.mean(mask*row)
            row *= mean/this_mean
        return arr
    elif axis is 'x':
        return arr_normalize_axis(arr.T,axis='y',mask=mask)
    else:
        raise Exception("invalid axis specification.")
        sys.exit(1)

# returns a secondary spectrum given a dynamic spectrum.
# Takes:
#    dyn - a dynamic spectrum numpy 2D array
#    subtract_secondary_background - whether or not to subtract the background
#        to improve contrast. Points without signal will have a mean of 0,
#        the points with meaningful data will have a mean higher than 0.
#    normalize_frequency - whether to try to get rid of stripeyness in the frequency axis
#    normalize_time - whether to try to get rid of stripeyness in the time axis
#    cut_off_bottom - whether to cut off the "mirror image" bottom half of the secondary spectrum
#    xscale - the multiplicative scale by which to cut down x
#    yscale - the multiplicative scale by which to cut down y
# Returns:
#    a numpy array containing the secondary spectrum
def get_secondary_spectrum(dyn,subtract_secondary_background=True,normalize_frequency=True,normalize_time=True,cut_off_bottom=True,xscale=1.,yscale=1.):
    dynamic = dyn - np.mean(dyn)
    secondary = np.fft.fftn(dynamic)
    #secondary /= secondary.max()
    secondary = 10.*np.log10(np.abs(np.fft.fftshift(secondary))**2) # in decibels?
    
    if normalize_frequency:
        mask = [1. if i<len(secondary[0])/4. or i>3*len(secondary[0])/4. else 0. for i in range(len(secondary[0]))]
        secondary = arr_normalize_axis(secondary,'y',mask)
    if normalize_time:
        mask = [1. if i<len(secondary)/4. else 0. for i in range(len(secondary))]
        secondary = arr_normalize_axis(secondary,'x',mask)
    if subtract_secondary_background:
        secondary_background = np.mean(secondary[:len(secondary)/4.][:len(secondary[0])/4.])
        secondary = secondary - secondary_background
    
    ysize = secondary.shape[0]
    xsize = secondary.shape[1]
    
    xmin = int(xsize/2. - xsize/(2.*xscale))
    xmax = int(xsize/2. + xsize/(2.*xscale))
    
    ymin = int(ysize/2. - ysize/(2.*yscale))
    
    if cut_off_bottom:
        ymax = int(ysize/2.)
    else:
        ymax = int(ysize/2. + ysize/(2.*yscale))
    
    return secondary[ymin:ymax,xmin:xmax]

# given a list of files, returns the dynamic and secondary spectra for both
def get_dyn_and_sec(files):
    dyn = [get_dynamic_spectrum(f) for f in files]
    sec = [get_secondary_spectrum(d) for d in dyn]
    return (dyn,sec)

# sorts a dictionary by key.
def sort_dict_by_key(dictionary):
    keys = []
    values = []
    for k,v in sorted(dictionary.items()):
        keys.append(k)
        values.append(v)
    return (keys,values)

# finds the axes that correspond to the secondary spectrum plot.
def get_sec_axes(filename):
    """
    Returns lists containing the values of the axis elements.
    Parameters: takes a list of filenames pointing to the relevant FITS files.
    Returns: a list of tuples ([conjugate frequency axis values],[conjugate time axis values])
    """
    
    

    hdulist = fits.open(filename)
    
    t_int = hdulist[0].header["T_INT"] #Gets time interval (delta t) from header
    nchunks = hdulist[0].header["NCHUNKS"] #number of time subintegrations
    BW = hdulist[0].header["BW"] #Gets bandwidth from header 
    nchans = hdulist[0].header["NCHANS"] #number of channels
    
    nyq_t = 1000. / (2. * t_int) #nyquist frequency for the delay axis of the secondary spectrum
    nyq_f = nchans / (2. * BW) #nyquist frequency for the fringe frequency axis of the secondary spectrum
    #print("nchunks:"+str(nchunks))
    #print("nchans:"+str(nchans))
    #print("nyq_t:"+str(nyq_t))
    #print("nyq_f:"+str(nyq_f))
    fringe = list(np.linspace(-nyq_t,nyq_t,nchunks))
    delay = list(reversed(np.linspace(0,nyq_f,nchans/2.)))
    #print("t: " + str(len(t_temp)))
    #print("f: " + str(len(f_temp)))
    return (delay,fringe)


"""
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
"""
# here's a pretty schweet class that keeps track of both a numpy array and its
# associated axes. It looks long, but it's actually really simple - .data contains
# the 2D numpy array with the data, and .x_axis and .y_axis hold the x and y axes
# respectively (.axes holds both as a tuple). It's useful because it does type checking
# and stuff for you, so instead of lugging around variables like dyn_data dyn_xaxis
# dyn_yaxis all the way through your program, you can just have an Indexed2D object
# that holds all the information for you, guaranteed to work in imshow or something.
#
# oh, another cool thing about this - you can access the values in the Indexed2D by
# the value of its axes. For instance, if you have an array that has 200 elements from
# -1 to 1 on the x axis, and 100 elements from 0 to 1 on the y axis, you can access the
# value of a point at y=-0.32 and x=0.57 by saying my_Indexed2D[-0.32,0.57]
class Indexed2D:
    def __init__(self,data=None,axes=None,dtype=float):
        self._is_data_set = False
        self._are_axes_set = False
        if data==None:
            self.data = np.array([[]])
        else:
            self.set_data(data,dtype)
        if axes==None:
            self.axes = ([],[])
            self.y_axis = []
            self.x_axis = []
        else:
            self.set_axes(axes)
        return 
    
    def __getitem__(self,tup):
        y = tup[0]
        x = tup[1]
        y_index = self.__get_y_index(y)
        x_index = self.__get_x_index(x)
        y_axis = self.y_axis[y_index]
        x_axis = self.x_axis[x_index]
        y_axis = y_axis if type(y_axis) == list else [y_axis]
        x_axis = x_axis if type(x_axis) == list else [x_axis]
        d = self.data[y_index,x_index]
        if len(np.shape(d)) <= 1:
            return d
        else:
            return Indexed2D(data=self.data[y_index,x_index],axes=(y_axis,x_axis))
    
    def __setitem__(self,tup,item):
        yval = tup[0]
        xval = tup[1]
        y_index = self.__get_y_index(yval)
        x_index = self.__get_x_index(xval)
        y_index = y_index if type(y_index) == list else [y_index]
        x_index = x_index if type(x_index) == list else [x_index]
        if len(np.shape(item)) == 0:
            item = np.array([np.array([item])])
        
        for y in y_index:
            for x in x_index:
                item_y = y - y_index[0]
                item_x = x - x_index[0]
                self.data[y,x] = item[item_y,item_x]
    
    def shape(self):
        y_shape = len(self.data)
        x_shape = len(self.data[0])
        return (y_shape, x_shape)
    
    def __get_y_index(self,value):
        if type(value)==slice:
            if value.start != None:
                if value.start<min(self.y_axis) or value.start>max(self.y_axis):
                    raise IndexError('y axis index out of bounds: ' + str(value.start))
                start_index = list(np.absolute([p - value.start for p in self.y_axis]))
                start_index = start_index.index(min(start_index))
            else:
                start_index = 0
            if value.stop != None:
                if value.stop<min(self.y_axis) or value.stop>max(self.y_axis):
                    raise IndexError('y axis index out of bounds: ' + str(value.stop))
                stop_index = list(np.absolute([p - value.stop for p in self.y_axis]))
                stop_index = stop_index.index(min(stop_index))
            else:
                stop_index = len(self.y_axis)-1
            if start_index<stop_index+1:
                return slice(start_index,stop_index+1)
            else:
                return slice(stop_index+1,start_index)
        else:
            index = [abs(p - value) for p in self.y_axis]
            index = index.index(min(index))
            return index
    
    def __get_x_index(self,value):
        if type(value)==slice:
            if value.start is not None:
                if value.start<min(self.x_axis) or value.start>max(self.x_axis):
                    raise IndexError('x axis index out of bounds: ' + str(value.start))
                start_index = list(np.absolute([p - value.start for p in self.x_axis]))
                start_index = start_index.index(min(start_index))
            else:
                start_index = 0
            if value.stop is not None:
                if value.stop<min(self.x_axis) or value.stop>max(self.x_axis):
                    raise IndexError('x axis index out of bounds: ' + str(value.stop))
                stop_index = list(np.absolute([p - value.stop for p in self.x_axis]))
                stop_index = stop_index.index(min(stop_index))
            else:
                stop_index = len(self.x_axis)-1
            if start_index<stop_index+1:
                return slice(start_index,stop_index+1)
            else:
                return slice(stop_index+1,start_index)
        else:
            index = [abs(p - value) for p in self.x_axis]
            index = index.index(min(index))
            return index
    
    def set_data(self,data,dtype=float):
        if type(data) is not list and type(data) is not np.ndarray:
            raise TypeError('Data does not have the right type.')
        if len(np.shape(data)) == 1:
            data = np.array([data])
        for d in data:
            if len(d)!=len(data[0]):
                raise IndexError('Data must be rectangular in shape.')
        for d in data:
            if type(d) is not list and type(d) is not np.ndarray:
                raise TypeError('Data does not have the right type.')
            for i in range(len(d)):
                if d[i] is not dtype:
                    try:
                        d[i] = dtype(d[i])
                    except Exception as e:
                        print('your data, type '+str(type(d[i]))+' could not be casted to '+str(dtype)+': ')
                        raise e
        if self._are_axes_set:
            y_axis_matching = len(data) == len(self.y_axis)
            x_axis_matching = len(data[0]) == len(self.x_axis)
            if y_axis_matching and x_axis_matching:
                self._is_data_set = True
                self.data = np.array(data)
            else:
                raise IndexError('Data must have dimensions as axes')
        else:
            self._is_data_set = True
            self.data = np.array(data)
        return
    
    def set_axes(self,axes):
        if type(axes) is not tuple:
            raise TypeError('Axes argument should be a tuple (y_axis,x_axis)')
        y_axis = axes[0]
        x_axis = axes[1]
        if type(y_axis) is not list and type(y_axis) is not np.ndarray:
            raise TypeError('The axes should be specified with a list or numpy array')
        if type(x_axis) is not list and type(x_axis) is not np.ndarray:
            raise TypeError('The axes should be specified with a list or numpy array')
        for i in range(len(y_axis)):
            if type(y_axis[i]) is not float:
                y_axis[i] = float(y_axis[i])
        for i in range(len(x_axis)):
            if type(x_axis[i]) is not float:
                x_axis[i] = float(x_axis[i])
        if self._is_data_set:
            y_axis_matching = len(y_axis) == len(self.data)
            x_axis_matching = len(x_axis) == len(self.data[0])
            if y_axis_matching and x_axis_matching:
                self._are_axes_set = True
                self.axes = (y_axis,x_axis)
                self.x_axis = x_axis
                self.y_axis = y_axis
            else:
                raise IndexError('Axes must have dimensions as data')
        else:
            self._are_axes_set = True
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.axes = (y_axis,x_axis)
        return
    
    def get_data(self):
        return np.array(self.data)
    
    def get_axes(self):
        return self.axes
    
    def get_x_axis(self):
        return self.x_axis
    
    def get_y_axis(self):
        return self.y_axis


# this class constructs, contains, and displays secondary spectra.
class Secondary():
    # initialize me with a filename
    def __init__(self,filename,hand=None,rotate=False):
        self.dynamic = get_dynamic_spectrum(filename,rotate=rotate)
        data = get_secondary_spectrum(self.dynamic)
        axes = get_sec_axes(filename)
        #print(type(axes[0]),type(axes[1]))
        self.sec = Indexed2D(data=data,axes=axes)
        self.hand=hand
        self.made_1D=False
        self.parabola_power = {}
        self.observation_name = os.path.basename(filename)
        self.band = self.observation_name.split("_")[1].split("M")[0]
    
    # the secondary object can be accessed like a list - sec[5,5] will return
    def __getitem__(self,value):
        return self.sec[value]
    
    def get(self,value):
        return self.sec.get_data().item(tuple(value))
    
    def linearize(self,nones=False):
        y_axis = self.get_y_axis()
        x_axis = self.get_x_axis()
        new_data = Indexed2D()
        y_sqrt = np.linspace(max(np.sqrt(y_axis)),0.,num=len(y_axis))
        new_data.set_axes((y_sqrt,x_axis))
        new_data.set_data(np.zeros((len(y_axis),len(x_axis))))
        for y in y_axis:
            for x in x_axis:
                new_data[np.sqrt(y),x] = self[y,x]
        if nones:
            for y in y_axis:
                for x in x_axis:
                    if new_data[y,x] == 0:
                        new_data[y,x] = None
        self.sec = new_data
    
    # gives the y axis of the secondary spectrum
    def get_y_axis(self):
        return self.sec.y_axis
    
    # gives the x axis of the secondary spectrum
    def get_x_axis(self):
        return self.sec.x_axis
    
    # crops the secondary spectrum by y_scale and x_scale percent
    def crop_percent(self,y_scale,x_scale):
        y_scale = float(y_scale)
        x_scale = float(x_scale)
        if y_scale<0 or x_scale<0 or y_scale>1 or x_scale>1:
            raise ValueError('x_scale and y_scale must be between 0 and 1.')
        y_max = max(self.sec.get_y_axis())
        x_max = max(self.sec.get_x_axis())
        self.sec = self.sec[y_max*y_scale:,-x_max*x_scale:x_max*x_scale]
        return
    
    # crops the secondary spectrum to the tuples specified by x_lim
    # and y_lim - i.e. if the secondary spectrum goes from -10 to 10
    # on the x axis and 0 to 5 on the y axis, you couls specify
    # crop( (-2.5,2.5) , (0,3) ) to crop the secondary spectrum
    # to those ranges.
    def crop(self,y_lim,x_lim):
        self.sec = self.sec[float(y_lim[0]):float(y_lim[1]),float(x_lim[0]):float(x_lim[1])]
    
    # gives sec as a numpy 2D array
    def get_sec(self):
        return self.sec.get_data()
    
    # plots the secondary spectrum to the current figure in matplotlib
    def show_sec(self):
        show_image(self.sec)
        if self.made_1D:
            self.overplot_parabolas([min(self.etas),max(self.etas)])
        plt.title(self.observation_name)
        plt.xlabel('delay')
        plt.ylabel('fringe frequency')
        return
    
    # plots parabolas over the secondary spectrum.
    # Takes:
    #    etas - a list of the curvatures of parabolas desired
    #    offsets - a list of the y-offsets desired for the parabolas
    # Returns:
    #    nothing, but plots the parabolas to the current matplotlib figure.
    def overplot_parabolas(self, etas, offsets = [0.]):
        for eta in etas:
            for offset in offsets:
                eta = float(eta)
                axis_x = self.get_x_axis()
                plot_x = [x+offset for x in axis_x]
                axis_y = self.get_y_axis()
                parab = []
                for x in axis_x:
                    y = eta*x**2 - eta*offset**2
                    parab.append(y)
                plt.plot(plot_x, parab, 'b-')
                plt.xlim((min(axis_x),max(axis_x)))
                plt.ylim((min(axis_y),max(axis_y)))
    
    # shows how much power is present in each eta value.
    # Requires make_1D_by_quadratic to have been run, and simply plots
    # the results from it to the current figure in matplotlib.
    def show_power_vs_eta(self,weird=False):
        if not self.made_1D:
            print("make_1D_by_quadratic has not been run yet")
            return

        if not weird:
            plt.plot(self.etas,self.powers)
            plt.xlabel("eta")
            plt.ylabel("Power(dB), arbitrary scaling")
            plt.title("Power vs eta, " + self.observation_name)
            #self.overplot_parabolas([min(sec.etas),max(sec.etas)])
            return
        else:
            fig = plt.figure()
            fig.subplots_adjust(bottom=0.2)
            plt.plot([1/eta**2 for eta in self.etas],self.powers)
            
            x_axis_points = np.linspace(1/max(self.etas)**2,1/min(self.etas)**2,10)
            x_axis = [round(1/np.sqrt(x),4) for x in x_axis_points]
            plt.xticks(x_axis_points,x_axis,rotation=90)
            plt.xlabel("eta")
            plt.ylabel("Power(dB), arbitrary scaling")
            plt.title("Power vs eta, " + self.observation_name)
            #self.overplot_parabolas([min(self.etas),max(self.etas)])
            return
    
    def __give_eta_list(self,eta_range,num_etas,decimal_places=4):
        if num_etas is not 1:
            x_max = np.sqrt(1/min(eta_range))
            x_min = np.sqrt(1/max(eta_range))
            return [1/x**2 for x in np.linspace(x_min,x_max,num_etas)]
        else:
            return [np.average(eta_range)]
    
    # finds the eta values of the parabolas in the secondary spectrum.
    # goes through the range of etas given to it, and determines the total power integrated
    # across the parabola defined by each eta. Since the parabolas are blurred out, we expect
    # to see a gaussian distribution in eta vs power.
    # Takes:
    #    eta_range - a tuple containing the minimum and maximum etas to explore
    #    num_etas - the number of etas to explore over the above range
    #    num_threads - the number of simultaneous processes to be used for multiprocessing
    #    sigma - idfk i forget, just leave it alone probably
    # Returns:
    #    a list of powers and their corresponding eta values. Returned in the form (etas,values)
    #    where both etas and values are lists.
    def make_1D_by_quadratic(self,eta_range,num_etas,num_threads=mp.cpu_count()-1,sigma=None):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        print(self.observation_name)
        
        etas = self.__give_eta_list(eta_range,num_etas)
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(crunchy, sec=self, hand=self.hand, sigma=sigma), etas)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        ret = sort_dict_by_key(powers)
        self.made_1D = True
        self.etas = ret[0]
        self.powers = ret[1]
        return ret
    
    ###### not fully debugged, use with caution ######
    #
    # finds the power along a parabola as a function of x. For instance, if all of the power in
    # a parabola is on the left side of the parabola and there is almost no power on the right side,
    # this function will show large values for x<0 and small values for x>0.
    def power_along_parabola(self,eta,num_arclets = 100,num_threads=mp.cpu_count()-1,sigma_px=3):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        eta = float(eta)
        max_x = np.sqrt(max(self.sec.get_y_axis())/eta)
        max_possible_x = np.absolute(max(self.sec.get_x_axis()))
        if max_x>max_possible_x:
            max_x = max_possible_x
        
        y_axis = self.get_y_axis()
        x_axis = self.get_x_axis()
        
        px_y = np.absolute(y_axis[1]-y_axis[0])
        px_x = np.absolute(x_axis[1]-x_axis[0])
        
        def dist_bw_pts(pt1,pt2):
            y1 = pt1[0]
            y2 = pt2[0]
            x1 = pt1[1]
            x2 = pt2[1]
            return np.sqrt( np.absolute(y1-y2)**2 + np.absolute(x1-x2)**2 )

        
        temp = [max_x*x**2 for x in np.linspace(0,1,num_arclets/2)]
        x_list = [-x for x in list(reversed(temp))[:-1]]
        x_list.extend(temp)
        y_list = [eta*x**2 for x in x_list]
        pts = [(y_list[i],x_list[i]) for i in range(len(x_list))]
        
        sigmas = []
        for i in range(len(pts)):
            if i == 0:
                sigmas.append( [np.absolute(pts[1][0]-pts[0][0]),np.absolute(pts[1][1]-pts[0][1])] )
            elif i == len(pts)-1:
                sigmas.append( [np.absolute(pts[-1][0]-pts[-2][0]),np.absolute(pts[-1][1]-pts[-2][1])] )
            else:
                sigma_y = px_y*sigma_px
                sigma_x = px_x*sigma_px
                sigmas.append( [sigma_y,sigma_x] )
        
        
        pts_and_sigmas = []
        for i in range(len(sigmas)):
            pts_and_sigmas.append( (pts[i],sigmas[i]) )
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(crunchy2, sec=self, hand=self.hand), pts_and_sigmas)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        self.parabola_power[eta] = powers
        return powers
        
    ####### not fully debugged, use with caution #######
    #
    # finds and returns the width of the parabola as a function of x or something.
    # IDK I'm kind of in a rush right now, just contact me somehow if you really want
    # to use this lol
    def parabola_width(self,eta,max_width,num_offsets,num_threads=mp.cpu_count()-1):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        print(self.observation_name)
        
        temp = [max_width*np.sqrt(x) for x in np.linspace(0,1,num_offsets/2)]
        offsets = [-x for x in list(reversed(temp))[:-1]]
        offsets.extend(temp)
        
        print(offsets)
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(crunchy3, sec=self, eta=eta), offsets)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        ret = sort_dict_by_key(powers)
        self.offsets = ret[0]
        self.offset_powers = ret[1]
        return ret