import numpy as np

def fftconvolve(imgA, imgB):
    """ calculates the convoluton of two input images.
    arguments:
        imgA - 1st image to convolve.
        imgB - 2nd image to convolve.  imgA and imgB must be the same shape.
    returns:
        Conv(imgA, imgB).  Shape is same as inputs.
    """
    from scipy.fftpack import fft2, ifft2

    assert isinstance(imgA, np.ndarray) and isinstance(imgB, np.ndarray), "Images must be arrays"
    assert imgA.shape == imgB.shape, "Images must be the same shape."
    assert imgA.ndim == 2, "Images must be two-dimensional."

    (ysize, xsize) = imgA.shape
    result = ifft2(fft2(imgA) * fft2(imgB))
    result = np.roll(result, int(ysize/2), axis=0)
    result = np.roll(result, int(xsize/2), axis=1)
    return result

def smooth_with_rectangle(img, boxsize):
    """Uses a convolution to average each pixel of an image by a surrounding box of boxsize pixels.
    arguments:
        img - input image to average over
        boxsize - size of box of which to average over. This can be a single number or a 2-tuple.
    returns:
        rectangle averaged img
    """
    from . import shape

    assert isinstance(img, np.ndarray), "Must be an array"
    assert img.ndim in (2, 3), "Must be two- or three- dimensional"

    if img.ndim == 3:
        imgshape = img[0].shape
    else:
        imgshape = img.shape

    if type(boxsize) in (tuple, list):
        assert len(boxsize) == 2, "boxsize must be a tuple of length 2."
        mask = shape.rect(imgshape, boxsize[0], boxsize[1])
    else:
        assert type(boxsize) in (float, int), "boxsize must be float or int"
        mask = shape.rect(imgshape, boxsize, boxsize)

    return _apply_smooth(img, mask)

def _apply_smooth(img, mask):
    """ Apply a smoothing mask to an image.  This is a helper function that encapsulates much of what each of these functions do once the mask is generated.
    arguments:
        img - image to smooth.  Can be 2d or 3d.
        mask - mask to smooth.  must be two dimensional, and the same shape as the x/y dimensions in img.
    returns:
        result - the smoothed array
    """
    assert isinstance(img, np.ndarray) and img.ndim in (2,3), "must be 2 or 3-dimensional array"
    if img.ndim == 3:
        assert mask.shape == img[0].shape, "image and mask must have the same x/y dimensions"
        result = np.zeros_like(img)
        masksum = mask.sum()
        for i in range(img.shape[0]):
            # normalize result by number of pixels in the mask
            result[i] = np.real(fftconvolve(img[i], mask))/(masksum)
        return result
    else:
        assert mask.shape == img.shape, "image and mask must have the same x/y dimensions"
        # normalize result by number of pixels in the mask
        return np.real(fftconvolve(img, mask))/(mask.sum())

def smooth_with_circle(img, radius):
    """Uses a convolution to average each pixel of an image by a surrounding circle of radius.
    arguments:
        img - input image to average over
        radius - radius of circle to average over
    returns:
        circularly-averaged img.
    """
    from . import shape

    assert isinstance(img, np.ndarray), "Must be an array"
    assert img.ndim == 2, "Must be two-dimensional"
    assert type(radius) in (float, int), "radius must be a float or int"

    (ys, xs) = img.shape
    return _apply_smooth(img, shape.circle(img.shape, radius, AA=False))

def smooth_with_gaussian(img, fwhm):
    """Uses a convolution to average each pixel of an image by a 2d gaussian of fwhm
    arguments:
        img - input image to average over
        fwhm - size of box of which to average over
    returns:
        img replaced by the gaussian function of fwhm.
    """
    from . import shape

    sigma_x = fwhm/(2*np.sqrt(2*np.log(2)))
    return _apply_smooth(img, shape.gaussian(img.shape, (sigma_x, sigma_x)))

def smooth_with_spline(img, nx, ny, order=3):
    """Smooth an image with a spline. It returns the fitted spline.
    parameters:
        img - Two-dimensional image.  If img is complex, abs(img) is calculated.
        nx - number of control points in x (columns).
        ny - number control points in y (rows).
        order - order of splines. Defaults to 3.
    returns:
        spline - The fit of the image with the same dimension as img.
    """
    from scipy.ndimage import map_coordinates
    assert img.ndim == 2, "image must be two dimensional"
    assert (type(nx), type(ny)) == (int,int), "nx and ny must be int"
    assert order in range(0,6), "Spline interpolation order must be betwen 0-5."
    order = int(order)
    j = complex(0,1)
    (ys, xs) = img.shape

    # reduce image down to (ny, nx) shape
    new_idx = np.mgrid[0:ys-1:ny*j, 0:xs-1:nx*j]

    # take absolute value if complex. map_coordinates doen't work
    # for some reason, map_coordinates crashes even if np.iscomplex().any() is false.
    img = abs(img)

    reduced = map_coordinates(img, new_idx, order=order)

    # blow image back up to img.shape
    (rys, rxs) = reduced.shape
    out_idx = np.mgrid[0:rys-1:ys*j, 0:rxs-1:xs*j]
    return map_coordinates(reduced, out_idx, order=order)
