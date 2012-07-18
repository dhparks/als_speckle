"""Library for input and output for coherent measurements at beamline 12.0.2 at
the Advanced Light Source.  This library handles reading and writing of arrays
and tabulated data into various image formats.  The standard format is FITS, but
the library handles .bmp, .jpg, .png as well.  Complex datatypes are also
supported with some functions.

Author: Keoki Seu (kaseu@lbl.gov)
Author: Daniel Parks (dhparks@lbl.gov)

"""
import numpy
from . import shape

# If you need to use the python open() function, use this!!
_open = open

# global overwrite option for FITS files
overwrite_default = False
def set_overwrite(val):
    """ set the overwrite default action for FITS files.

    arguments:
        val - boolean value weather FITS write functions should overwrite files
    returns:
        nothing, sets the variable overwrite_default
    """
    global overwrite_default
    if isinstance(val, bool):
        overwrite_default = val

#
############### Primary Wrappers ################
#
def open(filename, quiet=True, orientImageHDU=True, convert_to='float', delimiter='\t'):
    """Open a data file listed in filename. This function looks at the filename
    extension and passes the necessary arguments to the correct function:
    openfits, openimage, read_text_array, load_pickle, and open_ds9_mask.
    
    Recognized file extensions:
        images: jpg, jpeg, gif, png, bmp
        fits: fits
        text: txt, csv
        pickle: pck
        DS9 region: reg
        
    Returns:
        data from filename; see individual opener functions for details
    """
    
    # define extension types to control switching
    img_exts  = ('jpg', 'jpeg', 'gif', 'png', 'bmp', )
    fits_exts = ('fits', 'fit', )
    txt_exts  = ('txt', 'csv', )
    pck_exts  = ('pck', )
    ds9mask_exts = ('reg', )
    all_exts = img_exts + fits_exts + txt_exts + pck_exts + ds9mask_exts

    # get extension from filename
    assert len(filename.split('.')) >= 2, "filename appears to have no extension"
    ext = filename.split('.')[-1]
    assert ext in all_exts, "ext \"%s\" not recognized" % ext

    # pass arguments to correct opener
    if ext in img_exts:  return openimage(filename)
    if ext in fits_exts: return openfits(filename,quiet=quiet,orientImageHDU=True)
    if ext in txt_exts:  return read_text_array(filename,convert_to=convert_to,delimiter=delimiter)
    if ext in pck_exts:  return load_pickle(filename)
    if ext in ds9mask_exts: return open_ds9_mask(filename)

def save(filename,data,header={},components=['mag'],color_map='L',delimiter='\t',overwrite=None):
    """ Save components of an array as desired filetype specified by file
    extension. This is a wrapper to save_fits, save_image, write_text_array.
    
    arguments:
        filename - path where data will be saved
        data - ndarray to save at filename
        header - pyfits.Header object. Default is empty
        components - components of img to be saved. Must be supplied as a list.
            available components are: 'mag', 'phase', 'real', 'imag', 'polar',
            'cartesian'. Default is ['mag']
        color_map - If saving data as an image, the color map to use.  Options
            are 'L','A', 'B', 'SLS', 'HSV' and 'Rainbow'. Default is 'L'.
        delimiter - If saving data as a text file, the delimiter between data
            entries. Default is '\t' (tab)
        overwrite - Whether to overwrite a file already existing at filename.
            Default is False.

    returns:
        nothing. Will throw an exception if something wrong happens.
    """
    
    # define extension types to control switching
    img_exts  = ('jpg','jpeg','gif','png','bmp')
    fits_exts = ('fits',)
    txt_exts  = ('txt','csv')
    
    # get extension from filename
    assert len(filename.split('.')) >= 2, "filename appears to have no extension"
    ext = filename.split('.')[-1]
    assert ext in img_exts or ext in fits_exts or ext in txt_exts, "ext \"%s\" not recognized"%ext
    if ext in img_exts:  save_image(filename,data,components=components,color_map=color_map)
    if ext in fits_exts: save_fits(filename,data,header=header,components=components,overwrite=overwrite)
    if ext in txt_exts:  write_text_array(filename,data,header=header)

#
############### Text ########################
#
def write_text_array(filename, array, header='', delimiter='\t'):
    """ Write a tab-separated array to the filesystem.

    arguments:
        filename - filename to write to
        array - array name to write
        header - A string to write before the file is written. Defaults to empty
        delimiter - Delimiter to use. Defaults to tab ('\\t')
    returns:
        nothing.  It will throw an IOError if it cannot write the file
    """
    import csv
    with _open(filename, "w") as f:
        f.write(header)
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(array)

def read_text_array(filename, convert_to='float', delimiter='\t'):
    """ wraps csv to read a tab seperated file. Returns a list of each line.

    arguments:
        filename - filename to read
        convert_to - optional argument for what to convert the elements to.
            Can be 'float', 'int' or None.  If None, it will leave them as
            strings and return a list.  If 'float' or 'int', it will
            convert it to an array
        delimiter - delimiter to use. defaults to tab ('\t').
    returns:
        The parsed file as a list or array depending on convert_to
    """
    import csv
    assert convert_to in (None, 'float', 'int', 'complex'), "unknown type to convert, %s" % convert_to

    if convert_to == 'float':
        conv_fn = numpy.float
    elif convert_to == 'int':
        conv_fn = numpy.int
    else: # We should never get here.
        conv_fn = lambda x: x

    rows = []
    with _open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            for row in reader:
                rows.append(row)
        except csv.Error:
            pass
    if convert_to is not None:
        res = []
        for r in rows:
            try:
                res.append([conv_fn(d) for d in r])
            except ValueError:
                pass
        return numpy.array(res)
    else:
        return rows

#
############### FITS ########################
#

def openfits(filename, quiet=True, orientImageHDU=True):
    """ Open a FITS file. Uses pyfits. Tries to intelligently figure out where
    the data is located.  The LabView programs put it in the ImageHDU, and the
    Andor Solis program puts it in PrimaryHDU.

    arguments:
        filename - file to open.
        quiet - if we should try to be be informative when opening file.
            defaults to false
        orientImageHDU - If the data is in ImageHDU, there is a good chance it
            was written by LabView, which rotates and flips the image. If True,
            reorient the image back to the Andor coordinates.  Defaults to True.
    returns:
        img - the opened image data.
    """
    import pyfits
    import os

    assert os.path.exists(filename), "unknown file %s" % filename
    if not isinstance(quiet, bool):
        quiet = True
    
    def pr(msg):
        if not quiet:
            print msg

    # open fits file - Labview software stores data in ImageHDU and Andor stores it in PrimaryHDU.
    a = pyfits.open(filename)
    # return data where header shows NAXIS != 0
    if a[0].header['NAXIS'] != 0:
        pr('openfits: %s data located in PrimaryHDU' % filename)
        return a[0].data
    elif a[1].header['NAXIS'] != 0:
        pr('openfits: %s data located in ImageHDU' % filename)

        if orientImageHDU:
            # ImageHDU usually means that it was written by Labview. Default to reorient the image into the correct (andor) orientation.
            pr('openfits: orienting ImageHDU data properly')
            return _labview_to_andor(a[1].data)
        else:
            return a[1].data
    else:
        raise IOError('openfits: No idea where data is located!')

def openframe(filename, frame=0, quiet=True):
    """ Opens a single FITS frame.

    arguments:
        filename - name of file to open
        frame - optional argument of the frame to open.
            Defaults to 0 (first frame).
    returns:
        img - the opened image.
    """
    import pyfits
    a = pyfits.open(filename)
    if a[0].header['NAXIS'] == 3 and frame < a[0].header['NAXIS3']:
        if not quiet:
            print("openframe: %s is three dimensional, returning frame %d.\n" % (filename, frame))
        return a[0].section[frame,:,:]
    else:
        raise IOError("openframe: %s is not 3d or frame %d does not exist." % (filename, frame))

def openheader(filename, card=0):
    """ Open just the header from a filename.

    arguments:
        filename - file to open.
        card - optional argument of the header card. Defaults to 0th card.
    returns:
        header - the header in the file.
    """
    import pyfits
    return pyfits.getheader(filename, card)

def openimage(filename):
    """ Open an image file using PIL. This is currently restricted to
    greyscale images as the inverse colormap problem is difficult.
    
    arguments:
        filename - path to file
    returns:
        integer ndarray"""
    
    import Image
    import scipy.misc.pilutil as smp
    return smp.fromimage(Image.open(filename).convert("L"))

def writefits(filename, img, headerItems={}, overwrite=None):
    """ Write a FITS file,  optionally with a previously constructed header.

    arguments:
        filename - output filename to write.
        img - numpy array to write.
        headerItems - A dictionary of items to be added to the header.
            This will overwrite items if they are already in the header.
        overwrite - Overwrite the curernt file, if it exists.
    returns:
        returns nothing.  Will raise an error if it fails.
    """
    import pyfits
    assert isinstance(headerItems, dict), "headerItems must be a dictionary"
    if overwrite == None: overwrite = overwrite_default
    assert isinstance(overwrite,bool) or overwrite==None, "overwrite must be True/False"
    assert isinstance(img,numpy.ndarray), "array to save must be an array"

    header = pyfits.Header()

    # append items to header
    if isinstance(headerItems, dict):
        for key, val in headerItems.iteritems():
            #print "KEYTEST", key, "VAL", val
            if len(key) > 8 or key.rfind(' ') == 1:
                key = "HIERARCH " + key
            header.update(key, val)

    hdu = pyfits.PrimaryHDU(img, header=header)
    hdulist = pyfits.HDUList([hdu])
    try:
        hdulist.writeto(filename)
    except IOError as e:
        if overwrite:
            import os
            os.remove(filename)
            hdulist.writeto(filename)
        else:
            raise type(e)(e.message + " Try using the overwrite=True keyword to speckle.io.writefits()")

def write_complex_fits(base, fits, headerItems={}, overwrite=None):
    """ Write a complex array to disk.

    arguments:
        base - base filename; the function appends "_imag" and "_real".
        fits - Array to write.
        header - a pyfits.Header object. Defaults to empty.
        headerItems - A dictionary of items to be added to the header.
            This will overwrite items if they are already in the header.
        overwrite - Whether to overwrite file. Defaults to false.
    """

    writefits(base + "_imag.fits", numpy.imag(fits), headerItems, overwrite)
    writefits(base + "_real.fits", numpy.real(fits), headerItems, overwrite)

def write_mag_phase_fits(base, fits, header="", headerItems={}, overwrite=None):
    """ Write a complex array to disk by writing magnitude (abs())
    and phase (angle()).

    arguments:
        base - base filename; the function appends "_mag" and "_phase".
        fits - Array to write.
        header - a pyfits.Header object. Defaults to empty.
        headerItems - A dictionary of items to be added to the header.
            This will overwrite items if they are already in the header.
        overwrite - Whether to overwrite file. Defaults to false.
    returns:
        img - a complex-valued array
    """
    writefits(base + "_mag.fits", numpy.abs(fits), headerItems, overwrite)
    writefits(base + "_phase.fits", numpy.angle(fits), headerItems, overwrite)

def open_mag_phase_fits(base):
    """ Open a complex array that was written to disk using
    write_mag_phase_fits.

    arguments:
        base - base filename; The function tries to find
            base + _{mag,phase}.fits
    returns:
        img - a complex-valued array
    """
    return openfits( base + "_mag.fits")*numpy.exp(complex(0,1)*openfits(base + "_phase.fits"))
    
def open_complex_fits(base):
    """ Open a complex array that was written to disk using write_complex_fits.

    arguments:
        base - base filename; The function tries to find
            base + _{real,imag}.fits
    returns:
        img - a complex-valued array
    """
    return openfits(base + "_real.fits") + complex(0,1) * openfits( base + "_imag.fits" )

def _labview_to_andor(img):
    """Reorientes a LabView acquiried image so that it is the same orientation
    as what the Andor CCD camera software collects.

    arguments:
        img - image to rotate.
    returns:
        img - the aligned imaged rotated 90 degrees counter-clockwise and
            flipped horzontally.
    """
    assert img.ndim in (2,3), "_labview_to_andor: Image is neither 2D nor 3D."

    dim = img.ndim
    if dim == 3:
        (fr,ys,xs) = img.shape
        for f in range(fr):
            img[f] = numpy.rot90(numpy.fliplr(img[f]))
    elif dim == 2:
        img = numpy.rot90(numpy.fliplr(img))

    return img

def open_photon_counting_fits(filename, correct=False, sort=False, quiet=True):
    """Open a FITS file generated by the photon counting detector. The format of
    this file is a tuple of (x, y, gain, time_counter) and each has it's own
    datatype.

    arguments:
        filename - file to open
        correct - whether or not to correct overflows in the data
        sort - whether to sort the data by increasing incidence times
        quiet - whether to inform the user of what we're doing. Default is false
    returns:
        data - data as a numpy array. This will raise an IOError if the data is
            in an unfamiliar format.
    """
    import os
    assert os.path.exists(filename), "unknown file %s" % filename

    if not isinstance(correct, bool):
        correct = False

    if not isinstance(sort, bool):
        sort = False

    ncol = 4
    data = openfits(filename, quiet=quiet, orientImageHDU=False)

    if data.ndim == 2 and data.shape[1] == ncol:
        # data is good!
        pass
    elif data.ndim == 1:
        ys = len(data)
        newdata = numpy.zeros((ys, ncol), dtype='i4')
        for i in range(0, ncol):
            newdata[:, i] = data.field(i)
        data = newdata
    else:
        raise IOError("not sure of the data in %s." % filename)

    time_column = 3
    if correct:
        overflow = 2**31
        # Takes the data from the fast camera and corrects the overflow that happens when the counter rolls over.
        data = data.astype('float')
        timecol = data[:, time_column]
        diffdata = numpy.diff(timecol)
        
        finished = False
        while not finished:
            minarg = diffdata.argmin()
            if diffdata[minarg] < -2e9: # if you set this to 0, then there are false positives.  The data may have 'bumps' where the deriv < 0 but these are not rollovers.
                timecol[minarg+1:] +=  overflow
                diffdata[minarg] = 1
                print("Overflow at row %d. Shifted by %d" % (minarg, overflow))
            else:
                finished = True
        data[:, time_column] = timecol

    if sort:
        data = data[data[:, time_column].argsort(), :]

    return data

def save_fits(filename, img, header={}, components=['mag'], overwrite=None):
    """ Save components of an array as a fits file.
    
    arguments:
        filename - path where data will be saved
        img - ndarray to save at filename
        header - pyfits.Header object. Default is empty
        components - components of img to be saved. Must be supplied as a list.
            available components are:
                'mag', 'phase', 'real', 'imag', 'polar', 'cartesian'.
            Default is ['mag']
        overwrite - Whether to overwrite a file already existing at filename.
            Default is False.
    returns:
        nothing. Will throw an exception if something wrong happens.
    """

    exts = ['fits','jpg','gif','png','csv', 'jpeg','bmp']
    # remove any extension (if it exists)
    for ext in exts:
        filename = filename.replace("." + ext, "")

    for c in _process_components(components):
        writefits(filename + "_" + c + ".fits", _save_maps[c](img), header, overwrite)

def open_fits_header(filename, card=0):
    """    Open just the header from a filename.
        arguments:
            filename - file to open.
            card - optional argument of the header card. Defaults to 0th card.
        returns:
            header - the header in the file.
    """
    import pyfits
    return pyfits.getheader(filename, card)

def get_fits_binning(filename):
    """    get the Binning used in a FITS file.
        arguments:
            filename - the FITS file to open.
        returns:
            hbin, vbin - horizontal/vertical binning used.
    """
    try:
        hb = get_fits_key(filename, 'HBIN')
    except KeyError:
        hb  = 1

    try:
        vb = get_fits_key(filename, 'VBIN')
    except KeyError:
        vb  = 1

    return int(hb), int(vb)

def get_fits_exposure(filename):
    """    get the exposure time used in a FITS file.
        arguments:
            filename - the FITS file to open.
        returns:
            s - exposure time
    """
    try:
        s = get_fits_key(filename, 'EXPOSURE')
    except KeyError:
        s = 0.0
    return float(s)

def get_fits_accumulations(filename):
    """    get the # of accumulations used in a FITS file.
    
    arguments:
        filename - the FITS file to open.
            
    returns:
        # of accumulations. If it cannot find any it will return 1
    """
    try:
        acc = get_fits_key(filename, 'NUMACC')
    except KeyError:
        acc = 1

    return int(acc)

def get_fits_key(filename, key):
    """ Grabs a key from a FITS file. Searches cards for the key, and returns
    "no key" if it can't be found.
    
    arguments:
        filename - FITS file.
        key - key to find.
            
    returns:
        value of the key.  If none is found, returns "no key"
    """
    error = "Cannot find key"
    card = 0
    while True:
        try:
            hdr = open_fits_header(filename, card)
        except IndexError:
            raise KeyError(error)

        try:
            val = hdr[key]
        except KeyError:
            card += 1
            continue
        else:
            return val

    raise KeyError(error)

def get_fits_window(filename):
    """    Gets the window or region of interest for a given image.
    
        arguments:
            filename - the FITS file to open.
            
        returns:
            (xmin, xmax, ymin, ymax) - min/max values in the x and y directions
                of the window.
    """
    # returns the subimage window that was used when the image was taken.
    from string import split
    hdr=open_fits_header(filename)
    window = split(hdr['SUBRECT'], ',')
    # note out of order here, ymax comes before ymin
    (xmin, xmax, ymax, ymin) = (int(window[0]), int(window[1]), int(window[2]), int(window[3]))
    return (xmin, xmax, ymin, ymax)

def get_fits_kct(filename):
    """ Gets the kinetic cycle time from the header.
    """
    hdr = open_fits_header(filename)
    return hdr["KCT"]

def get_fits_dimensions(filename):
    """ Gets the dimenions of the FITS file from the header.
    
        arguments:
            filename - the FITS file to open.
            
        returns:
            (dim) - a list of the dimensions. The format is the same as the
                img.ndim command.
    """
    hdr = open_fits_header(filename)
    naxes = hdr["NAXIS"]
    dim = []
    while(naxes != 0):
        dim.append(hdr["NAXIS%d" % naxes])
        naxes -= 1

    return tuple(dim)

def get_fits_acq_time(filename, timezone="US/Pacific"):
    """Tries to figure out the acquisition time of a FITS file by looking for
    the date/time via the header.  The function looks in three places for the
    time (each acquistion program we use stores it in a different location).
    This function requires the pytz python library.

    Andor Solis camera:
        hdu[0].header['DATE']    = '2010-01-24T18:14:49' / file creation date
        hdu[0].header['FRAME']   = '2010-01-24T18:14:49.000' / Start of frame

    Labview acquisition:
        hdu[0].header['DATETIME']= '2010/01/24 10:26:25' / Date and time

    SSI camera:
        hdu[1].header['DATE']    = '2011-05-05'         / Date of header
        hdu[1].header['DATE-OBS']= '2011-05-05'         / Date of data
        hdu[1].header['TIME']    = '00:26:00'           / Time of data

    arguments:
        filename - File to find acquisition time.
        timezone - Timezone where the data was acquired. Default is "US/Pacific"

    returns:
        datetime - a TZ localized datetime object of acquisiton time. If it
            cannot be found, it returns False
    """
    import datetime
    try:
        import pytz
    except ImportError:
        print "get_acq_time: Cannot import the pytz module.  Please install it."
        return False

    assert timezone in pytz.common_timezones, "%s is not a timezone" % timezone
    tz = pytz.timezone(timezone)
    utc = pytz.UTC

    try:
        # SSI camera date. This is given in UTC
        fitsdate = get_fits_key(filename, "DATE") + " " + get_fits_key(filename, "TIME")
        parsed_date = datetime.datetime.strptime(fitsdate, "%Y-%m-%d %H:%M:%S")
        return utc.localize(parsed_date)
    except KeyError:
        pass

    try:
        # Labview Date
        fitsdate = get_fits_key(filename, "DATETIME")
        parsed_date = datetime.datetime.strptime(fitsdate, "%Y/%m/%d %H:%M:%S")
        return tz.localize(parsed_date)
    except KeyError:
        pass

    try:
        # Andor Date. This is given in UTC.
        fitsdate = get_fits_key(filename, "DATE")
        parsed_date = datetime.datetime.strptime(fitsdate, "%Y-%m-%dT%H:%M:%S")
        return utc.localize(parsed_date)
    except KeyError:
        pass
    
    # We weren't able to parse the date/time.
    return False

#
############### pickles #####################
#
def load_pickle(filename):
    """ Load a pickled file.
    
    arguments:
        filename - File to pickle.  This should be a pickled file.
        
    returns:
        the data in the pickle file
    """
    import pickle
    with _open(filename,'rb') as f:
        return pickle.load(f)
    
def save_pickle(path,data):
    """ Load a pickled file.
    
    arguments:
        filename - File to pickle.  This should be a pickled file.
        This will overwrite the existing file if it exists.
        
    returns:
        no return value.  It throws an error if it is not successful.
    """
    import pickle
    with _open(path,'wb') as f:
        pickle.dump(data,f)

#
############### PNG #########################
#
    
def save_image(filename, img, components=['mag'], color_map='L'):
    """ Save components of an array as an image using PIL.  The type of image
    depends on the extension of the filename.

    arguments:
        filename - filename to save
        img - image to save
        components - a list/string/set/tuple of components to save.  The
            function will append the name to the end and save the component.
        color_map - color map to use.  Options are 'A', 'B', 'SLS', 'HSV'
            and 'Rainbow'.
            
    returns:
        nothing. Will throw an exception if something wrong happens.
    """
    
    # check to see if an image extension has been included.
    # jpg, png, gif, bmp
    # png is default if no extension is included.
    
    ext = filename.split('.')[-1]

    exts = ['jpg','gif','png', 'jpeg']
    if ext in exts:
        filename = filename.replace("." + ext, "")
    else:
        ext = 'png'

    for c in _process_components(components):
        write_image(filename + "_" + c + "." + ext, _save_maps[c](img), color_map) 

def write_image(filename, array, color_map):
    """Write an image to disk as an image.  The image type depends on the
    extension.

    arguments:
        filename - filename to save
        array - array to save.
        color_map -  Color map to use.  Can be generated by color_maps()
        
    returns:
        no return arguments.
    """

    import scipy.misc.pilutil as smp

    im = smp.toimage(array)

    ext = filename.split('.')[-1]
    assert ext in ('gif', 'png', 'jpg', 'jpeg'), "unknown file format, %s" % ext
    
    if color_map == None or color_map == 'L':
         # greyscale
        smp.toimage(array).save(filename)
    else:
        im.putpalette(color_maps(color_map))
        # now save the image
        if ext in ('gif','png'):  im.save(filename)
        if ext in ('jpg','jpeg'): im.convert('RGB').save(filename,quality=95)

def color_maps(map):
    """ List of colormaps.  Used for image output.

    arguments:
        map - map to use, based on DS9 colormaps.
            Available: ('A', 'B', 'SLS', 'HSV', 'Rainbow')
        
    returns:
        R, G, B - RGB components
    """
    assert map in ('A', 'B', 'SLS', 'HSV', 'Rainbow'), "unknown map: %s" % map

    x = numpy.arange(256)
    R,G,B = numpy.zeros_like(x),numpy.zeros_like(x),numpy.zeros_like(x)
    
    if map == 'B':
        # define the color palette in terms of RGB channels. these are piece-wise functions.
        R[128:] = 255
        R[65:128] = 4*x[65:128]-256
        
        G[192:] = 255
        G[128:192] = 4*x[128:192]-4*128
        
        B[0:64] = 4*x[0:64]
        B[64:128] = 255-4*x[64:128]+4*64
        B[192:256] = 4*x[192:256]-4*192
        
    if map == 'A':
        # define the color palette in terms of RGB channels
        R[128:] = 255
        R[65:128] = 4*x[65:128]-256
        
        G[0:64] = 4*x[0:64]
        G[64:128] = 255-4*x[64:128]+4*64
        G[192:256] = 4*x[192:256]-4*192
        
        B[32:128] = 255./(128-32)*x[32:128]-32*255./(128-32)
        B[128:192] = 255-4*x[128:192]+4*128
        
    if map == 'SLS':
        # define the color palette in terms of RGB channels
        R = [0.0, 2.89608407e+00, 1.28814192e+01, 2.16174221e+01, 3.05164814e+01, 3.71828384e+01, 4.65450516e+01, 5.54545746e+01, 6.43763809e+01, 7.35827255e+01, 8.01909561e+01, 8.93354721e+01, 9.82664032e+01, 1.07176178e+02, 1.16540070e+02, 1.23334236e+02, 1.31731171e+02, 1.29592331e+02, 1.26456703e+02, 1.22984734e+02, 1.20545601e+02, 1.18364738e+02, 1.15136818e+02, 1.11859810e+02, 1.08433487e+02, 1.05778107e+02, 1.03739746e+02, 1.00581406e+02, 9.72177429e+01, 9.40934677e+01, 9.20555038e+01, 8.95245819e+01, 8.46622238e+01, 7.81553574e+01, 7.27032776e+01, 6.78973160e+01, 6.12927361e+01, 5.51178818e+01, 4.92835846e+01, 4.43443184e+01, 3.89126053e+01, 3.25472565e+01, 2.69479771e+01, 2.10111504e+01, 1.48540335e+01, 1.04002934e+01, 4.10968018e+00, 0.0, 1.00339182e-01, 2.29601515e-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.63066635e-02, 0.0, 1.45226109e+00, 8.57071972e+00, 1.49110279e+01, 2.21798592e+01, 2.93516369e+01, 3.43640900e+01, 4.14933739e+01, 4.79902954e+01, 5.54719048e+01, 6.07938156e+01, 6.71726227e+01, 7.46045609e+01, 8.11540604e+01, 8.85470810e+01, 9.41761017e+01, 1.00240929e+02, 1.06787376e+02, 1.13938438e+02, 1.21306511e+02, 1.26484932e+02, 1.33157990e+02, 1.40075287e+02, 1.46494766e+02, 1.53567413e+02, 1.59965149e+02, 1.65308685e+02, 1.72734009e+02, 1.79802139e+02, 1.86417206e+02, 1.92191315e+02, 1.98382782e+02, 2.03986572e+02, 2.07059967e+02, 2.11263367e+02, 2.14367920e+02, 2.16783066e+02, 2.20544128e+02, 2.24637680e+02, 2.27490921e+02, 2.30710709e+02, 2.34646286e+02, 2.37798721e+02, 2.41123764e+02, 2.45157394e+02, 2.47674896e+02, 2.51487900e+02, 2.54980820e+02, 2.53101135e+02, 2.52245102e+02, 2.51667236e+02, 2.50819962e+02, 2.50009537e+02, 2.49199387e+02, 2.48352921e+02, 2.47749908e+02, 2.46931625e+02, 2.46117325e+02, 2.45306747e+02, 2.44446976e+02, 2.43828506e+02, 2.43109100e+02, 2.41227188e+02, 2.42091141e+02, 2.43545181e+02, 2.44077667e+02, 2.44836426e+02, 2.45671219e+02, 2.46486038e+02, 2.47339371e+02, 2.48019409e+02, 2.48720566e+02, 2.49568130e+02, 2.50383438e+02, 2.51225250e+02, 2.51962860e+02, 2.52512009e+02, 2.54012833e+02, 2.54896484e+02, 2.53910294e+02, 2.54082489e+02, 2.53451050e+02, 2.52957291e+02, 2.52884705e+02, 2.52010925e+02, 2.51985489e+02, 2.52047638e+02, 2.51794678e+02, 2.50963776e+02, 2.51050171e+02, 2.50253143e+02, 2.49963303e+02, 2.49895111e+02, 2.49029022e+02, 2.44431641e+02, 2.39266724e+02, 2.35645370e+02, 2.30791321e+02, 2.25876083e+02, 2.21056290e+02, 2.15595062e+02, 2.09888290e+02, 2.05900757e+02, 2.01676758e+02, 1.96591217e+02, 1.91699219e+02, 1.86634842e+02, 1.82603043e+02, 1.76865295e+02, 1.77383041e+02, 1.84176300e+02, 1.89690521e+02, 1.94145477e+02, 1.98560760e+02, 2.04218445e+02, 2.09157349e+02, 2.14943207e+02, 2.20625076e+02, 2.24402847e+02, 2.29737656e+02, 2.35436325e+02, 2.40234375e+02, 2.45905731e+02, 2.50350525e+02, 2.55000000e+02, 2.54920029e+02, 2.54998215e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999985e+02]
        G = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.92219685e-02, 0.0, 7.24638522e-01, 5.13517094e+00, 9.03740406e+00, 1.20978966e+01, 1.63778191e+01, 2.04285088e+01, 2.44919968e+01, 2.86254673e+01, 3.16316338e+01, 3.58354797e+01, 3.98886871e+01, 4.39389381e+01, 4.81810608e+01, 5.12025757e+01, 5.53063126e+01, 5.92537308e+01, 6.53793030e+01, 7.23345642e+01, 7.72787857e+01, 8.35368881e+01, 8.93004379e+01, 9.55702515e+01, 1.02499161e+02, 1.07675774e+02, 1.13581474e+02, 1.20272530e+02, 1.26791443e+02, 1.33602859e+02, 1.39120392e+02, 1.43866699e+02, 1.50449707e+02, 1.56707245e+02, 1.62300797e+02, 1.66626465e+02, 1.70500427e+02, 1.76160370e+02, 1.81661880e+02, 1.86612625e+02, 1.92104492e+02, 1.95964111e+02, 2.01066589e+02, 2.06089508e+02, 2.11869949e+02, 2.16638565e+02, 2.20715637e+02, 2.26620209e+02, 2.30927994e+02, 2.29182907e+02, 2.28362442e+02, 2.26952942e+02, 2.25761246e+02, 2.25082825e+02, 2.23499451e+02, 2.21774109e+02, 2.20509003e+02, 2.19913101e+02, 2.18281296e+02, 2.17374405e+02, 2.16113831e+02, 2.14340622e+02, 2.12977692e+02, 2.12243515e+02, 2.11952148e+02, 2.12010742e+02, 2.11985657e+02, 2.12064423e+02, 2.11343933e+02, 2.10934418e+02, 2.11014664e+02, 2.11000000e+02, 2.10999969e+02, 2.10999939e+02, 2.10999969e+02, 2.10999985e+02, 2.11000031e+02, 2.11009583e+02, 2.10963486e+02, 2.11170547e+02, 2.12108368e+02, 2.13792755e+02, 2.14634766e+02, 2.16022293e+02, 2.17329605e+02, 2.18900131e+02, 2.19696335e+02, 2.21211502e+02, 2.22280563e+02, 2.23023727e+02, 2.24734619e+02, 2.25609039e+02, 2.26957474e+02, 2.28275467e+02, 2.29845291e+02, 2.30031921e+02, 2.29991547e+02, 2.29999985e+02, 2.29999969e+02, 2.30000000e+02, 2.30000000e+02, 2.29999969e+02, 2.29999969e+02, 2.30000000e+02, 2.29999985e+02, 2.29999939e+02, 2.29999985e+02, 2.30000031e+02, 2.29968216e+02, 2.30131653e+02, 2.29247055e+02, 2.27677078e+02, 2.26955460e+02, 2.26147751e+02, 2.25596054e+02, 2.24409866e+02, 2.22807785e+02, 2.22117004e+02, 2.21459503e+02, 2.20644409e+02, 2.19633484e+02, 2.17967529e+02, 2.17182968e+02, 2.16552322e+02, 2.15781052e+02, 2.14596100e+02, 2.10409973e+02, 2.06329132e+02, 2.03283997e+02, 1.99231873e+02, 1.96036743e+02, 1.92001373e+02, 1.87762451e+02, 1.84750031e+02, 1.80667877e+02, 1.76541382e+02, 1.73275803e+02, 1.69296127e+02, 1.66140839e+02, 1.62249863e+02, 1.58050735e+02, 1.54674622e+02, 1.50825439e+02, 1.47518173e+02, 1.43835022e+02, 1.39568771e+02, 1.36057968e+02, 1.32377960e+02, 1.28885971e+02, 1.25397316e+02, 1.21167641e+02, 1.17045448e+02, 1.13035049e+02, 1.10275848e+02, 1.06924194e+02, 1.02804390e+02, 9.79604645e+01, 9.10108795e+01, 8.40896759e+01, 7.89611435e+01, 7.21371002e+01, 6.54759827e+01, 5.80564270e+01, 5.17206230e+01, 4.63700447e+01, 3.88736916e+01, 3.24732933e+01, 2.60165844e+01, 1.85788918e+01, 1.34715366e+01, 6.84745979e+00, 3.32012564e-01, 0.0, 2.38011591e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.51287663e-01, 0.0, 8.47359562e+00, 2.71742725e+01, 4.42804756e+01, 6.03119621e+01, 7.34783707e+01, 9.19307327e+01, 1.09714386e+02, 1.26840912e+02, 1.43588104e+02, 1.56865753e+02, 1.75463058e+02, 1.92384109e+02, 2.09463455e+02, 2.26722992e+02, 2.40322174e+02, 2.55000000e+02, 2.54748688e+02, 2.54994293e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999985e+02]
        B = [0.0, 3.42408895e+00, 1.52247782e+01, 2.55354328e+01, 3.69006004e+01, 4.49421959e+01, 5.61787834e+01, 6.75640945e+01, 7.80262909e+01, 8.96791000e+01, 9.78080673e+01, 1.08649521e+02, 1.20102539e+02, 1.30603470e+02, 1.42326462e+02, 1.50806198e+02, 1.60803345e+02, 1.62733505e+02, 1.63572357e+02, 1.64948425e+02, 1.66203262e+02, 1.66891312e+02, 1.67645950e+02, 1.69086151e+02, 1.70465164e+02, 1.71031021e+02, 1.72629852e+02, 1.73581665e+02, 1.74843079e+02, 1.76362701e+02, 1.76931931e+02, 1.78339828e+02, 1.79959381e+02, 1.82011429e+02, 1.84135178e+02, 1.85414429e+02, 1.87706650e+02, 1.90255234e+02, 1.92557434e+02, 1.93844803e+02, 1.95960251e+02, 1.97974518e+02, 1.99840805e+02, 2.02320541e+02, 2.03700943e+02, 2.05547806e+02, 2.08147964e+02, 2.10573914e+02, 2.13065567e+02, 2.16227905e+02, 2.18675980e+02, 2.22118195e+02, 2.25282532e+02, 2.27663193e+02, 2.30882263e+02, 2.33308197e+02, 2.36668686e+02, 2.39917099e+02, 2.42312454e+02, 2.45507553e+02, 2.47968109e+02, 2.51177780e+02, 2.54608673e+02, 2.55000000e+02, 2.54977478e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.54999985e+02, 2.55000000e+02, 2.54897049e+02, 2.55000000e+02, 2.52802109e+02, 2.46566940e+02, 2.42373596e+02, 2.38499603e+02, 2.32843765e+02, 2.27322113e+02, 2.22430649e+02, 2.17900833e+02, 2.13533554e+02, 2.07824997e+02, 2.03048386e+02, 1.98111725e+02, 1.93367706e+02, 1.89252121e+02, 1.83475266e+02, 1.78643112e+02, 1.72906281e+02, 1.67233276e+02, 1.61222092e+02, 1.56881516e+02, 1.51402573e+02, 1.46463135e+02, 1.41006424e+02, 1.34937027e+02, 1.30408707e+02, 1.25241226e+02, 1.19386551e+02, 1.13682571e+02, 1.07723335e+02, 1.02890373e+02, 9.88256912e+01, 9.31640244e+01, 8.73550262e+01, 8.16683273e+01, 7.74545135e+01, 7.27596588e+01, 6.72614899e+01, 6.21860352e+01, 5.63796501e+01, 5.19075546e+01, 4.75276413e+01, 4.18541641e+01, 3.69274292e+01, 3.11300602e+01, 2.64091377e+01, 2.20556774e+01, 1.73827114e+01, 1.69932270e+01, 1.52735481e+01, 1.35621319e+01, 1.24330368e+01, 1.17578945e+01, 1.00828381e+01, 9.30368900e+00, 7.78850985e+00, 6.71948195e+00, 5.97627783e+00, 4.26542234e+00, 3.39553237e+00, 2.02441716e+00, 7.94162571e-01, 9.44109485e-02, 0.0, 4.19447524e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.58877429e-02, 0.0, 3.56573910e-01, 1.25203204e+00, 2.06054139e+00, 2.83971667e+00, 3.46013188e+00, 4.31802416e+00, 5.14298630e+00, 5.89353752e+00, 6.53256321e+00, 7.39289522e+00, 8.20530796e+00, 9.02634525e+00, 9.81676102e+00, 1.04310284e+01, 1.12880850e+01, 1.20983677e+01, 1.29116726e+01, 1.37350950e+01, 1.43368855e+01, 1.51800432e+01, 1.59904575e+01, 1.67905807e+01, 1.76924286e+01, 1.80452538e+01, 1.80416813e+01, 1.88875027e+01, 1.96942177e+01, 2.05515881e+01, 2.11694202e+01, 2.19063206e+01, 2.29364243e+01, 2.19358158e+01, 2.00753326e+01, 1.88134995e+01, 1.73271847e+01, 1.56574955e+01, 1.40278387e+01, 1.23211451e+01, 1.09612284e+01, 9.55891705e+00, 7.86372185e+00, 6.23310375e+00, 4.54952002e+00, 3.12460232e+00, 1.73184884e+00, 4.25788611e-01, 0.0, 1.69107970e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.51287663e-01, 0.0, 8.47359562e+00, 2.71742725e+01, 4.42804756e+01, 6.03119621e+01, 7.34783707e+01, 9.19307327e+01, 1.09714386e+02, 1.26840912e+02, 1.43588104e+02, 1.56865753e+02, 1.75463058e+02, 1.92384109e+02, 2.09463455e+02, 2.26722992e+02, 2.40322174e+02, 2.55000000e+02, 2.54748688e+02, 2.54994293e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999985e+02]

    if map == 'HSV':
        R = [2.72020960e+00, 3.87280388e+01, 5.14365463e+01, 5.73812447e+01, 6.23493080e+01, 6.61366730e+01, 6.92822189e+01, 7.25153046e+01, 7.53159561e+01, 7.74525757e+01, 7.89026871e+01, 7.94784775e+01, 8.06972885e+01, 8.22111130e+01, 8.29590225e+01, 8.29216690e+01, 8.35770798e+01, 8.38119049e+01, 8.29359131e+01, 8.29931412e+01, 8.30746384e+01, 8.25450897e+01, 8.19252701e+01, 8.20024719e+01, 8.20650406e+01, 8.13450851e+01, 8.09345551e+01, 8.10200348e+01, 8.01424255e+01, 8.00217056e+01, 7.94850769e+01, 7.86036835e+01, 7.79703751e+01, 7.72918015e+01, 7.64749603e+01, 7.56849518e+01, 7.48969345e+01, 7.41126328e+01, 7.33243408e+01, 7.25313950e+01, 7.17394333e+01, 7.09528732e+01, 7.01523514e+01, 6.94483337e+01, 6.81954117e+01, 6.67237167e+01, 6.60121002e+01, 6.52880020e+01, 6.43186188e+01, 6.27675476e+01, 6.20504990e+01, 6.13418999e+01, 5.99373207e+01, 5.86239128e+01, 5.78472595e+01, 5.61752853e+01, 5.52937584e+01, 5.45789337e+01, 5.35272598e+01, 5.19067497e+01, 5.12341843e+01, 4.97073097e+01, 4.85540771e+01, 4.76369743e+01, 4.60539589e+01, 4.55146027e+01, 4.42803383e+01, 4.27658081e+01, 4.20932426e+01, 4.04601631e+01, 3.94820633e+01, 3.84151306e+01, 3.68405075e+01, 3.61000061e+01, 3.53931427e+01, 3.40555229e+01, 3.26663361e+01, 3.19370232e+01, 3.02781219e+01, 2.93175087e+01, 2.87845688e+01, 2.79893703e+01, 2.63555756e+01, 2.53736038e+01, 2.46574421e+01, 2.37173672e+01, 2.20695934e+01, 2.12533627e+01, 2.04667702e+01, 1.96934872e+01, 1.88190842e+01, 1.71640644e+01, 1.62988491e+01, 1.55264254e+01, 1.47301226e+01, 1.39408865e+01, 1.31754999e+01, 1.25904016e+01, 1.17987776e+01, 1.08472977e+01, 9.98529243e+00, 9.22431087e+00, 8.61732197e+00, 7.93847036e+00, 8.02348804e+00, 7.34120035e+00, 6.14478016e+00, 7.25139999e+00, 1.16799145e+01, 1.55176315e+01, 1.93288860e+01, 2.40642357e+01, 2.75353374e+01, 3.16634178e+01, 3.63256111e+01, 4.07172279e+01, 4.57683411e+01, 4.94445648e+01, 5.37821655e+01, 5.79367256e+01, 6.22382545e+01, 6.70666656e+01, 7.17309418e+01, 7.72749481e+01, 8.22695236e+01, 8.69738007e+01, 9.17236786e+01, 9.63738861e+01, 1.01898338e+02, 1.06912460e+02, 1.11857841e+02, 1.17445419e+02, 1.22944061e+02, 1.27958374e+02, 1.31550751e+02, 1.37144318e+02, 1.42693787e+02, 1.47536545e+02, 1.52643433e+02, 1.58258789e+02, 1.63737122e+02, 1.69241470e+02, 1.74779266e+02, 1.80354904e+02, 1.85779617e+02, 1.90399490e+02, 1.95845337e+02, 2.01390808e+02, 2.07038208e+02, 2.12085022e+02, 2.12571411e+02, 2.13485062e+02, 2.14056259e+02, 2.14005875e+02, 2.14793716e+02, 2.15665680e+02, 2.16031006e+02, 2.16110443e+02, 2.17002274e+02, 2.16931091e+02, 2.17559601e+02, 2.18054138e+02, 2.18070740e+02, 2.18904480e+02, 2.19765289e+02, 2.20037048e+02, 2.20032944e+02, 2.20930511e+02, 2.20948730e+02, 2.21427887e+02, 2.22069702e+02, 2.22007507e+02, 2.22797134e+02, 2.23669067e+02, 2.24030228e+02, 2.24113449e+02, 2.25004166e+02, 2.24930847e+02, 2.25563751e+02, 2.26052887e+02, 2.26064240e+02, 2.26962250e+02, 2.26961304e+02, 2.27244797e+02, 2.28030518e+02, 2.28030640e+02, 2.28916168e+02, 2.28983414e+02, 2.29175949e+02, 2.30029556e+02, 2.30880676e+02, 2.30962234e+02, 2.31360184e+02, 2.32070572e+02, 2.31966080e+02, 2.32827545e+02, 2.32981842e+02, 2.33290833e+02, 2.34064880e+02, 2.33946075e+02, 2.34765884e+02, 2.35001389e+02, 2.35224915e+02, 2.36055420e+02, 2.35918655e+02, 2.36440643e+02, 2.37069153e+02, 2.36995834e+02, 2.37886536e+02, 2.37962524e+02, 2.38364304e+02, 2.39070770e+02, 2.38967422e+02, 2.39830917e+02, 2.39980743e+02, 2.40294052e+02, 2.41057159e+02, 2.40999390e+02, 2.40946472e+02, 2.41252487e+02, 2.42055786e+02, 2.41936264e+02, 2.42716980e+02, 2.43016006e+02, 2.43179214e+02, 2.44036606e+02, 2.43929657e+02, 2.44647873e+02, 2.45034561e+02, 2.45122513e+02, 2.46009628e+02, 2.45930130e+02, 2.46565506e+02, 2.47095139e+02, 2.46877792e+02, 2.46105896e+02, 2.45521011e+02, 2.44687103e+02, 2.43918121e+02, 2.43068481e+02, 2.43054810e+02, 2.42557861e+02, 2.41925644e+02, 2.42031738e+02, 2.41933853e+02, 2.42653015e+02, 2.43034897e+02, 2.43127335e+02, 2.44013870e+02, 2.44686447e+02, 2.46168686e+02, 2.47250198e+02, 2.47900726e+02, 2.49425934e+02, 2.51040100e+02, 2.52762161e+02]
        G = [2.66006351e+00, 3.78167725e+01, 5.04032860e+01, 5.64303703e+01, 6.11502991e+01, 6.40974426e+01, 6.73521729e+01, 6.98670273e+01, 7.22496338e+01, 7.44675064e+01, 7.59027023e+01, 7.64784851e+01, 7.77028656e+01, 7.91876221e+01, 8.00358582e+01, 8.09831009e+01, 8.09529724e+01, 8.12597046e+01, 8.20470581e+01, 8.19900665e+01, 8.20095444e+01, 8.19576035e+01, 8.21955872e+01, 8.30609360e+01, 8.38211517e+01, 8.46457520e+01, 8.52177505e+01, 8.59852905e+01, 8.68473053e+01, 8.78207169e+01, 8.84913177e+01, 8.97059097e+01, 9.10897827e+01, 9.17072830e+01, 9.24736099e+01, 9.35654373e+01, 9.52121277e+01, 9.68257217e+01, 9.76287766e+01, 9.88800659e+01, 1.00533890e+02, 1.02108658e+02, 1.03599266e+02, 1.05908325e+02, 1.07860626e+02, 1.09546768e+02, 1.11993210e+02, 1.13496597e+02, 1.14999786e+02, 1.17516281e+02, 1.19839401e+02, 1.22199539e+02, 1.24580132e+02, 1.26938019e+02, 1.29392853e+02, 1.32555786e+02, 1.35789932e+02, 1.38423691e+02, 1.40979858e+02, 1.44218369e+02, 1.47355881e+02, 1.50420715e+02, 1.54095154e+02, 1.57769623e+02, 1.59883774e+02, 1.60486664e+02, 1.61716904e+02, 1.63228851e+02, 1.63935455e+02, 1.64736267e+02, 1.65528336e+02, 1.66321259e+02, 1.67109451e+02, 1.67893677e+02, 1.68671555e+02, 1.69521515e+02, 1.70054398e+02, 1.70023544e+02, 1.70836441e+02, 1.71654861e+02, 1.72223373e+02, 1.73010651e+02, 1.73794144e+02, 1.74584946e+02, 1.75378281e+02, 1.76168152e+02, 1.76952866e+02, 1.77739426e+02, 1.78526917e+02, 1.79342148e+02, 1.80044296e+02, 1.79957962e+02, 1.80668808e+02, 1.81481995e+02, 1.82269928e+02, 1.83059128e+02, 1.83815826e+02, 1.84448563e+02, 1.85031372e+02, 1.85141724e+02, 1.86015671e+02, 1.86777206e+02, 1.87382675e+02, 1.88061554e+02, 1.87976501e+02, 1.88735809e+02, 1.89537048e+02, 1.90327026e+02, 1.91105911e+02, 1.91947174e+02, 1.91943741e+02, 1.92462875e+02, 1.93104111e+02, 1.93919662e+02, 1.94936203e+02, 1.94950577e+02, 1.95396683e+02, 1.96075302e+02, 1.96855011e+02, 1.96974182e+02, 1.97313980e+02, 1.98185730e+02, 1.98983826e+02, 1.98944672e+02, 1.99493073e+02, 2.00339890e+02, 2.01109344e+02, 2.01949356e+02, 2.01950577e+02, 2.02426361e+02, 2.03297882e+02, 2.04030334e+02, 2.03941025e+02, 2.04635071e+02, 2.05239487e+02, 2.06015503e+02, 2.05944153e+02, 2.06554077e+02, 2.07424652e+02, 2.08054565e+02, 2.07962036e+02, 2.08817398e+02, 2.08990005e+02, 2.09259277e+02, 2.10129532e+02, 2.10951492e+02, 2.10950058e+02, 2.11429123e+02, 2.12225250e+02, 2.13283203e+02, 2.09144913e+02, 2.04731461e+02, 2.00656143e+02, 1.95860886e+02, 1.92037491e+02, 1.87382492e+02, 1.83008789e+02, 1.78985825e+02, 1.74186371e+02, 1.70235214e+02, 1.66279236e+02, 1.62336441e+02, 1.58294846e+02, 1.53503036e+02, 1.49536194e+02, 1.45383179e+02, 1.42408813e+02, 1.38661484e+02, 1.35423752e+02, 1.31826950e+02, 1.27788773e+02, 1.23885025e+02, 1.20831497e+02, 1.16970352e+02, 1.13357216e+02, 1.10279808e+02, 1.07124657e+02, 1.03979973e+02, 1.00813179e+02, 9.76428299e+01, 9.44699173e+01, 9.14367142e+01, 8.95737610e+01, 8.70955505e+01, 8.47716370e+01, 8.11920013e+01, 7.83959122e+01, 7.67234268e+01, 7.44365311e+01, 7.29500351e+01, 7.05517273e+01, 6.85940247e+01, 6.81612778e+01, 6.95021515e+01, 7.19465561e+01, 7.42743988e+01, 7.66358490e+01, 7.91149673e+01, 8.10320892e+01, 8.31905365e+01, 8.62225800e+01, 8.84564133e+01, 9.09367523e+01, 9.29651566e+01, 9.48231735e+01, 9.78120575e+01, 1.00809013e+02, 1.03091743e+02, 1.05381081e+02, 1.08464149e+02, 1.11234802e+02, 1.13690063e+02, 1.16933472e+02, 1.19214058e+02, 1.22230560e+02, 1.25089844e+02, 1.27482559e+02, 1.30656738e+02, 1.33937943e+02, 1.35893951e+02, 1.38339752e+02, 1.41507675e+02, 1.44644073e+02, 1.47868652e+02, 1.50800049e+02, 1.53115753e+02, 1.56280579e+02, 1.59438889e+02, 1.62611343e+02, 1.65773346e+02, 1.68913696e+02, 1.72057526e+02, 1.75228897e+02, 1.78371521e+02, 1.81677353e+02, 1.85483231e+02, 1.87927567e+02, 1.91249008e+02, 1.94373932e+02, 1.97511246e+02, 2.00675110e+02, 2.03793869e+02, 2.07200607e+02, 2.11173325e+02, 2.14301178e+02, 2.17379852e+02, 2.21001343e+02, 2.24855072e+02, 2.27927475e+02, 2.30965134e+02, 2.34941574e+02, 2.37763000e+02, 2.40685684e+02, 2.44639893e+02, 2.48158691e+02, 2.51304657e+02]
        B = [2.73238683e+00, 3.86760712e+01, 5.21746140e+01, 5.83531303e+01, 6.37280960e+01, 6.92189331e+01, 7.31034622e+01, 7.71374588e+01, 8.06896210e+01, 8.38270645e+01, 8.67649841e+01, 8.85576935e+01, 9.14515762e+01, 9.44625244e+01, 9.70923157e+01, 9.87620392e+01, 1.00431404e+02, 1.02976639e+02, 1.05240982e+02, 1.06706497e+02, 1.09063599e+02, 1.10988731e+02, 1.12502083e+02, 1.14087883e+02, 1.15642319e+02, 1.17291504e+02, 1.18435478e+02, 1.19970581e+02, 1.21678123e+02, 1.23686005e+02, 1.24522911e+02, 1.25703247e+02, 1.27075493e+02, 1.28404053e+02, 1.30105576e+02, 1.31363174e+02, 1.32147110e+02, 1.33835938e+02, 1.34619080e+02, 1.35925674e+02, 1.37328201e+02, 1.38052567e+02, 1.39717377e+02, 1.40657410e+02, 1.41373566e+02, 1.42355606e+02, 1.43989410e+02, 1.44784561e+02, 1.45317505e+02, 1.46278152e+02, 1.47919922e+02, 1.48741348e+02, 1.49526749e+02, 1.50301422e+02, 1.51169708e+02, 1.52824722e+02, 1.53696075e+02, 1.54467255e+02, 1.55263672e+02, 1.56050308e+02, 1.56834991e+02, 1.57624786e+02, 1.58416504e+02, 1.59160797e+02, 1.60154266e+02, 1.58364120e+02, 1.55821259e+02, 1.53616699e+02, 1.52126541e+02, 1.49786209e+02, 1.47468842e+02, 1.44762100e+02, 1.41611191e+02, 1.39361572e+02, 1.36299622e+02, 1.33050003e+02, 1.30145065e+02, 1.27801865e+02, 1.24734489e+02, 1.20718643e+02, 1.18041855e+02, 1.14955795e+02, 1.11879303e+02, 1.08115349e+02, 1.04072830e+02, 1.00273727e+02, 9.71962280e+01, 9.33849182e+01, 8.87992477e+01, 8.43039017e+01, 8.04536896e+01, 7.64983673e+01, 7.26500015e+01, 6.81372452e+01, 6.35841751e+01, 5.96766663e+01, 5.50470009e+01, 5.15415382e+01, 4.67926559e+01, 4.10837898e+01, 3.59117546e+01, 3.13064690e+01, 2.78745708e+01, 2.29269638e+01, 1.82636681e+01, 1.36089439e+01, 8.53270912e+00, 5.33268976e+00, 5.03895664e+00, 5.04124975e+00, 4.33817339e+00, 3.46709490e+00, 2.94619322e+00, 2.98642612e+00, 2.08728027e+00, 2.04583287e+00, 1.60074413e+00, 9.28172529e-01, 1.00137377e+00, 1.06632507e+00, 6.49357021e-01, 0.0, 1.46725504e-02, 1.48585523e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.05565523e-03, 0.0, 5.26859984e-03, 8.59378278e-01, 1.04002631e+00, 9.62853253e-01, 1.12302315e+00, 2.00039530e+00, 1.94447243e+00, 2.49195004e+00, 3.36280608e+00, 4.04643536e+00, 3.96043372e+00, 4.67969704e+00, 5.48787928e+00, 6.30203724e+00, 6.97661543e+00, 7.62226248e+00, 8.44874191e+00, 9.23563194e+00, 1.00215816e+01, 1.08066635e+01, 1.15971708e+01, 1.23917341e+01, 1.31843576e+01, 1.39439516e+01, 1.55594950e+01, 1.65917873e+01, 1.73071098e+01, 1.82036552e+01, 1.98751602e+01, 2.06323853e+01, 2.19948177e+01, 2.32408142e+01, 2.47696609e+01, 2.56124649e+01, 2.68336868e+01, 2.84910603e+01, 3.00495529e+01, 3.16196156e+01, 3.32004814e+01, 3.47872658e+01, 3.63677979e+01, 3.79376678e+01, 3.95100327e+01, 4.10984459e+01, 4.26579742e+01, 4.43544350e+01, 4.67004700e+01, 4.79566650e+01, 4.95974007e+01, 5.11679268e+01, 5.38607597e+01, 5.56760406e+01, 5.72585373e+01, 5.95689964e+01, 6.10651894e+01, 6.34876289e+01, 6.56432190e+01, 6.91363373e+01, 7.44345016e+01, 8.08253937e+01, 8.62374649e+01, 9.23797836e+01, 9.86748505e+01, 1.03058327e+02, 1.08111244e+02, 1.15186012e+02, 1.20398109e+02, 1.26173653e+02, 1.30935425e+02, 1.34711960e+02, 1.40186844e+02, 1.45539108e+02, 1.50182861e+02, 1.55767776e+02, 1.60669037e+02, 1.65390228e+02, 1.70138702e+02, 1.74868561e+02, 1.78736557e+02, 1.83330948e+02, 1.87802979e+02, 1.91658081e+02, 1.95570129e+02, 1.99605591e+02, 2.02432587e+02, 2.06467758e+02, 2.10427750e+02, 2.13599014e+02, 2.17261063e+02, 2.21071960e+02, 2.24105911e+02, 2.26402878e+02, 2.29400162e+02, 2.32649506e+02, 2.35633560e+02, 2.37919769e+02, 2.40279434e+02, 2.42718094e+02, 2.44783051e+02, 2.46172195e+02, 2.46856827e+02, 2.47532715e+02, 2.48057022e+02, 2.48036880e+02, 2.48940979e+02, 2.48945572e+02, 2.49444565e+02, 2.50068619e+02, 2.49999603e+02, 2.50877014e+02, 2.51037125e+02, 2.50960022e+02, 2.51140640e+02, 2.52011368e+02, 2.51929855e+02, 2.52575684e+02, 2.53079254e+02, 2.52949646e+02, 2.53784134e+02, 2.54010910e+02, 2.54196777e+02]

    if map == 'Rainbow':
        R = [2.52756851e+02, 2.47920288e+02, 2.42838226e+02, 2.37954803e+02, 2.33013474e+02, 2.28072144e+02, 2.23130661e+02, 2.18189133e+02, 2.13247650e+02, 2.08306122e+02, 2.03364639e+02, 1.98424835e+02, 1.93468491e+02, 1.88831833e+02, 1.83455292e+02, 1.78103485e+02, 1.73230392e+02, 1.68279312e+02, 1.63309967e+02, 1.58485306e+02, 1.53730179e+02, 1.48751556e+02, 1.43817795e+02, 1.38876541e+02, 1.33935242e+02, 1.28993958e+02, 1.24043480e+02, 1.19157921e+02, 1.13970009e+02, 1.08674751e+02, 1.03801094e+02, 9.88463135e+01, 9.39047546e+01, 8.89631805e+01, 8.40215759e+01, 7.90799942e+01, 7.41431580e+01, 6.91718597e+01, 6.43566818e+01, 5.95929985e+01, 5.46109657e+01, 4.96903496e+01, 4.46994019e+01, 3.91385956e+01, 3.40655022e+01, 2.93925571e+01, 2.44846764e+01, 1.95337830e+01, 1.45930109e+01, 9.68643284e+00, 4.60923147e+00, 3.35283130e-01, 0.0, 2.49895602e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.92476954e-02, 0.0, 6.92567706e-01, 5.33760834e+00, 1.03941078e+01, 1.60112495e+01, 2.09646053e+01, 2.59179363e+01, 3.07963200e+01, 3.55119820e+01, 4.04868927e+01, 4.54208031e+01, 5.03622475e+01, 5.53036308e+01, 6.02449188e+01, 6.51861725e+01, 7.01278152e+01, 7.50654221e+01, 8.00098877e+01, 8.53017960e+01, 9.04168320e+01, 9.53260117e+01, 1.00275101e+02, 1.05216484e+02, 1.10157944e+02, 1.15097427e+02, 1.20057373e+02, 1.24925468e+02, 1.29647171e+02, 1.34623657e+02, 1.39557327e+02, 1.44513031e+02, 1.49366089e+02, 1.54744019e+02, 1.59960907e+02, 1.64728836e+02, 1.69708054e+02, 1.74646683e+02, 1.79588043e+02, 1.84529327e+02, 1.89470657e+02, 1.94411972e+02, 1.99353241e+02, 2.04294586e+02, 2.09236008e+02, 2.14184158e+02, 2.19097244e+02, 2.24171783e+02, 2.29483032e+02, 2.34450470e+02, 2.39382736e+02, 2.44302902e+02, 2.49306488e+02, 2.53935959e+02, 2.55000000e+02, 2.54958527e+02, 2.54999985e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.54999985e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02]
        G = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.90175269e-02, 0.0, 1.72289237e-01, 4.04783487e+00, 9.15578175e+00, 1.40563030e+01, 1.90081558e+01, 2.39044437e+01, 2.93102226e+01, 3.43433037e+01, 3.92619553e+01, 4.42057762e+01, 4.91469994e+01, 5.40828056e+01, 5.90461998e+01, 6.37428932e+01, 6.86540604e+01, 7.36028519e+01, 7.85437393e+01, 8.34851379e+01, 8.84316711e+01, 9.33509827e+01, 9.87755356e+01, 1.03762459e+02, 1.08692619e+02, 1.13635323e+02, 1.18576836e+02, 1.23518326e+02, 1.28459793e+02, 1.33401306e+02, 1.38342682e+02, 1.43284058e+02, 1.48225342e+02, 1.53161652e+02, 1.58127060e+02, 1.62794418e+02, 1.67823074e+02, 1.73452911e+02, 1.78364670e+02, 1.83333481e+02, 1.88183228e+02, 1.92918335e+02, 1.97896759e+02, 2.02830322e+02, 2.07771820e+02, 2.12713394e+02, 2.17654892e+02, 2.22596344e+02, 2.27544388e+02, 2.32440399e+02, 2.37573730e+02, 2.42908020e+02, 2.47764725e+02, 2.52813934e+02, 2.55000000e+02, 2.54965714e+02, 2.54996506e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999939e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.54999969e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.54999985e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999939e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.54999969e+02, 2.55000000e+02, 2.54999969e+02, 2.54970444e+02, 2.55000000e+02, 2.54576202e+02, 2.50176376e+02, 2.45128448e+02, 2.40216507e+02, 2.35275131e+02, 2.30333694e+02, 2.25392380e+02, 2.20437927e+02, 2.15557251e+02, 2.10183884e+02, 2.05096512e+02, 2.00190735e+02, 1.95244995e+02, 1.90303604e+02, 1.85362274e+02, 1.80420837e+02, 1.75479309e+02, 1.70537872e+02, 1.65604126e+02, 1.60627823e+02, 1.55907028e+02, 1.51026138e+02, 1.46125137e+02, 1.40673676e+02, 1.35392776e+02, 1.30546844e+02, 1.25834679e+02, 1.20866104e+02, 1.15931389e+02, 1.10990181e+02, 1.06049026e+02, 1.01107819e+02, 9.61666260e+01, 9.12253494e+01, 8.62840576e+01, 8.13404388e+01, 7.64195557e+01, 7.13952789e+01, 6.59851379e+01, 6.10853119e+01, 5.61342278e+01, 5.11926727e+01, 4.62511063e+01, 4.13104858e+01, 3.63570518e+01, 3.14628277e+01, 2.67576618e+01, 2.17874184e+01, 1.69097538e+01, 1.18383865e+01, 6.95705032e+00]
        B = [2.54999985e+02, 2.54999985e+02, 2.54999985e+02, 2.54999939e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.54999969e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.54999985e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999954e+02, 2.54999939e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.54999969e+02, 2.55000000e+02, 2.54999969e+02, 2.54999985e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999969e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.55000000e+02, 2.54999985e+02, 2.54999969e+02, 2.54988144e+02, 2.54998718e+02, 2.54976929e+02, 2.51465057e+02, 2.46327393e+02, 2.41431335e+02, 2.36490143e+02, 2.31552139e+02, 2.26586884e+02, 2.21743103e+02, 2.17003784e+02, 2.12025162e+02, 2.07075531e+02, 2.02227875e+02, 1.96781525e+02, 1.91655014e+02, 1.86854996e+02, 1.81883804e+02, 1.76943680e+02, 1.72002121e+02, 1.67060532e+02, 1.62118942e+02, 1.57177444e+02, 1.52236038e+02, 1.47294632e+02, 1.42353287e+02, 1.37398911e+02, 1.32517258e+02, 1.27140648e+02, 1.22058189e+02, 1.17151360e+02, 1.12205772e+02, 1.07264481e+02, 1.02330383e+02, 9.73578873e+01, 9.26484375e+01, 8.77613754e+01, 8.28060837e+01, 7.78657608e+01, 7.29242249e+01, 6.79747467e+01, 6.30664864e+01, 5.79212303e+01, 5.26477394e+01, 4.77158508e+01, 4.27747078e+01, 3.78333969e+01, 3.28921814e+01, 2.79509888e+01, 2.30098019e+01, 1.80686169e+01, 1.31275387e+01, 8.24500751e+00, 3.04505062e+00, 0.0, 3.42398882e-02, 6.79758471e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # zip RGBs then flatten to single list rather than list of lists
    return [item for sublist in zip(R,G,B) for item in sublist]

_save_maps = {
    "mag": numpy.abs,
    "phase": numpy.angle, 
    "real": numpy.real,
    "imag": numpy.imag,
}

def _process_components(components):
    """ parse the components array and try to figure out what components
    are to be saved.
    
    arguments:
        components - component string, list,set or tuple.
        
    returns:
        components - a list of components to save.
    """

    if type(components) == str:
        assert components in _save_maps.keys() or \
            components in ('polar', 'cartesian', 'all'), \
            "components %s not recognized" % components
        #print components
        components = [components]
    elif type(components) in (list,tuple,set):
        components = list(components)

    # shortcuts to save common combinations of components
    if 'polar' in components:
        components.extend( ('mag', 'phase') )
        components.remove( 'polar' )
    if 'cartesian' in components:
        components.extend( ['real', 'imag'] )
        components.remove('cartesian')
    if 'all' in components:
        components = _save_maps.keys()
    #print components, "components"
    for elem in components:
        assert elem in _save_maps.keys(), "unknown component %s" % elem

    # remove duplicates
    return list(set(components))
#
############### DS9 Masks #########################
#
def _draw_polygon(shapedesc, dim):
    from PIL import Image, ImageDraw
    ys, xs = dim
    
    nvertex = len(shapedesc)/2
    verticies = []
    for v in range(nvertex):
        xc = shapedesc[v*2]
        yc = shapedesc[v*2+1]
        verticies.append((xc,yc))

    img = Image.new("L", (xs, ys), 0)
    ImageDraw.Draw(img).polygon(verticies, outline=1, fill=1)

    return numpy.array(img)

def open_ds9_mask(filename, intersectionsRemovedFromMask = False):
    """ From a region file input construct a mask.
    
    arguments:
        file - region filename
        intersectionsRemovedFromMask - Whether intersected regions should count
            as part of the mask. Defaults to false.

    returns:
        mask - binary mask of the regions
    """
    shapes = []
    with _open(filename, "r") as f:
        aline = f.readline()
        while aline:
            if aline[0] == "#":
                if aline.find("Filename:") != -1:
                    file = aline[len("# Filename:"):-1]
                    try:
                        dim = get_fits_dimensions(file)
                    except IOError:
                        dim = (1, 2048, 2048)
    
                aline = f.readline()
                continue
    
            elif aline.split("(")[0] in ("box", "circle", "polygon", "ellipse", "annulus"):
                shapes.append(aline[0:-1])
            aline = f.readline()

    if len(dim) == 3: # cut down to 2d, ignore 3rd dimension
        dim = dim[1:]
    
    data = numpy.zeros(dim)
    (ys, xs) = dim
    for s in shapes:
        shapetype, shapedesc = s.split("(")
        shapest = s.find("(")
        shapeend = s.find(")")
        shapedesc = [float(val) for val in s[shapest+1:shapeend].split(",") ]
        if shapetype == "circle":
             # fix discrepancy between ds9 index and python idex (1 vs 0)
            shapedesc[0] = shapedesc[0] - 1
            shapedesc[1] = shapedesc[1] - 1

            xc, yc, rad = shapedesc
            data += shape.circle( dim, rad, (yc,xc), AA=False )
        elif shapetype == "box":
             # fix discrepancy between ds9 index and python idex (1 vs 0)
            shapedesc[0] = shapedesc[0] - 1
            shapedesc[1] = shapedesc[1] - 1

            if len(shapedesc) == 7:
                xc, yc, in_xw, in_yw, out_xw, out_yw, angle = shapedesc
                angle = -angle # ds9 angles are measured in the opposite direction.
                data += shape.rect( (ys,xs), (out_yw, out_xw), (yc, xc) ) - shape.rect( (ys,xs), (in_yw, in_xw), (yc, xc))
            else:
                xc, yc, xw, yw, angle = shapedesc
                angle = -angle # ds9 angles are measured in the opposite direction.
                data += shape.rect( (ys,xs), (yw, xw), (yc, xc))

        elif shapetype == "polygon":
             # fix discrepancy between ds9 index and python idex (1 vs 0)
            for i in range(len(shapedesc)):
                shapedesc[i] = shapedesc[i] - 1

            data += _draw_polygon(shapedesc, dim)
        elif shapetype == "ellipse":
             # fix discrepancy between ds9 index and python idex (1 vs 0)
            shapedesc[0] = shapedesc[0] - 1
            shapedesc[1] = shapedesc[1] - 1
            if len(shapedesc) == 7:
                xc, yc, in_xw, in_yw, out_xw, out_yw, angle = shapedesc
                angle = -angle # ds9 angles are measured in the opposite direction.
                data += shape.ellipse( (ys,xs), (out_yw, out_xw), (yc, xc), angle, AA=False) - shape.ellipse( (ys,xs), (in_yw, in_xw), (yc, xc), angle, AA=False)
            else:
                xc, yc, xw, yw, angle = shapedesc
                angle = -angle # ds9 angles are measured in the opposite direction.
                data += shape.ellipse( (ys,xs), (yw, xw), (yc, xc), angle, AA=False)
    
        elif shapetype == "annulus":
             # fix discrepancy between ds9 index and python idex (1 vs 0)
            shapedesc[0] = shapedesc[0] - 1
            shapedesc[1] = shapedesc[1] - 1

            (xc, yc, rin, rout) = shapedesc
            data += shape.annulus((ys, xs), (rin, rout), (yc, xc), AA=False)

    if intersectionsRemovedFromMask:
        return numpy.where( (data % 2) == 0 , 0, 1)
    else:
        return numpy.where(data >= 1, 1, 0)
