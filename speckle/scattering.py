"""A Library for calculating various scattering parameters such as Flangosaurus
distances, Q, and energy to wavelength conversion.

Author: Keoki Seu (KASeu@lbl.gov)
"""
#scattering
#    qmaps
#    intensity patterns ( cl?, e?)
#    propagation
import numpy as np

# Constants for each component between the scattering center of the chamber and camera.
bl_length_components = {
    "valveLen": 2 * 25.4, # manual valve 2", spec from mdc
    "CCDLen": (2 + 0.78) * 25.4 - 2, # 2" nipple and 0.78" flange (spec, mdc) that area always attached to camera. According to Andor focal plane of camera is 2.0mm +- 0.4mm behind flange
    "BeamBlockDiodeLen": 4.92 * 25.4, # Beam block and Diode T flange: 4.92" spec from mdc
    "CSCToFlange": 259.4, # CSC numbers from Tom, according to model
    "CSCToThruFlange": 368.3 + 0.78*25.4, # the 0.78" is for the 6" to 2.75" reducing flange on the xmission port.  0.78" spec from mdc
}

def calculateQ( imgshape, center, theta, energy, camera_distance, calctype='Qr', pixel_pitch = 0.0135, thetaOffset=0, rotateAngle=0, transmission=False):
    """Given an imput image and center, calculate Q.
    
    arguments:
        imgshape - image size in (xsize, ysize)
        center - center coordinate in (xcenter, ycenter) format
        theta - Sample angle where Q needs to be calculated.
        energy - Beam energy in eV.
        camera_distance - Sample to CCD length, in mm. A dictionary of ditances can be found in bl_length_components[].
        pixel_pitch - pixel size, in mm/px.  If the image is binned multiply by binning. Defaults to 0.0135 (13.5 um pixels, binned 1)
        calctype - Specify the type of calculation to do. Options are 'Qx', 'Qy', 'Qz', 'Qr', 'Q'.  Defaults to 'Qr'
        thetaOffset - This is the angle where theta=0.  It is a peak in the theta scan of the surface.  This is used to correct for theta. Defaults to 0.0.
        rotateAngle - Rotate the image in the plane by a given angle. Sometimes the camera is not perfectly aligned with the scattering plane, so it may be necessary to rotate Q slightly to compensate.  This is seen in RPM, FeF2, HoTiO data, among others. Defaults to 0.0.
    returns:
        calculated q vector of dimension imgshape. The units of the qmap is A^{-1}
    """
    assert type(theta) in (int, float) and theta > 0, "theta must be greater than 0"
    assert type(energy) in (int, float) and energy > 0, "energy must be greater than 0"
    assert type(camera_distance) in (int, float) and camera_distance > 0, "camera_distance must be greater than 0"
    assert type(pixel_pitch) in (int, float) and pixel_pitch > 0, "pixel_pitch must be greater than 0"
    assert len(imgshape) == 2, "image is not 2d"
    assert len(center) == 2, "center is not a pair"

    if calctype not in ('Q', 'Qx', 'Qy' ,'Qz', 'Qr'):
        print("calculateQ: invalid calctype %s. Defaulting to Qr.\n" % calctype)
        calctype = 'Qr'

    # the location of the specular in two-theta.
    thetaSpecular = 2.0 * (theta - thetaOffset) * (np.pi/180.0)
#    print("calculateQ: theta - %0.2f degrees.\nthetaOffset - %0.2f degrees\n2theta - %0.2f degrees.\ncamera_distance - %0.1f mm\nenergy - %0.1f eV" % (theta, thetaOffset, thetaSpecular*180/np.pi, camera_distance, energy))

    # Image parameters
    xs, ys = imgshape
    cimgx, cimgy = center
#    print("calculateQ: xsize=%d, ysize=%d; xcenter=%d, ycenter=%d" % (xs, ys, cimgx, cimgy))

    # set up camera coordinates. The coordinates are sample coordinates meaning x = manipulator z, y = manipulator x, z= manipulator y.  This is done so that 
    xr = np.ones(xs,dtype=float)
    yr = np.ones(ys,dtype=float)

    # set up the x,y,z coordinates. right handed system has Qx along beam, + Qx downstream; Qy out of scattering plane, + is towards loadlock; and Qz, + is perpendicular to sample surface
    x = np.ones(ys,dtype=float)*camera_distance
    y = np.arange(0,xs,1,dtype=float)
    z = np.arange(0,ys,1,dtype=float)
    # Translate CCD coordinates so most intense peak is at (y,z) = 0
    y = y - cimgx
    z = z - cimgy

    # convert coordinates to mm
    # x is already in mm
    y = y*pixel_pitch
    z = z*pixel_pitch

    # get our xyz coordinate matricies
    X = np.outer(x,xr)
    Y = np.outer(yr,y)
    Z = np.outer(z,xr)
    
    # calculate the distance to camera and normalize coordinates
    camdistance = np.sqrt(X*X + Y*Y + Z*Z)
    X = X/camdistance
    Y = Y/camdistance
    Z = Z/camdistance

    k = energy_to_wavevector(energy)

    # rotate to new angle. Rotate along Qy for twotheta, and rotate along Qx for rotateAngle
#    print("calculateQ: 2theta/2 = %0.2f degrees, rotateAngle = %0.2f degrees, %s is in A^-1." % ((thetaSpecular/2)*(180/np.pi), rotateAngle, calctype))
    # this rotates counter-clockwise about center of peak on CCD
    if rotateAngle != 0:
        Y, Z = _rotateX(Y, Z, rotateAngle*np.pi/180)

    # in transmission the coordinates change slightly, X->Z, Z->X.
    # The reason my calculation does not work for both reflection and transmission is that I assume that sample angle = detector 2theta.  This is not true for transmission (sample angle =90, detector =0)
    if transmission:
#        print("calculating for transmission")
        (X, Z) = (Z, X)
        (kx_i, ky_i, kz_i) = (0, 0, k)
    else:
        (kx_i, ky_i, kz_i) = (k*np.cos(thetaSpecular/2), 0, -k*np.sin(thetaSpecular/2))

    # this rotates to correct 2theta angle
    if thetaSpecular != 0:
        X, Z = _rotateY(X, Z, thetaSpecular/2)

    Qx = k*X - kx_i
    Qy = k*Y - ky_i
    Qz = k*Z - kz_i

    Q = np.sqrt(Qx*Qx + Qy*Qy + Qz*Qz)
    Qr = np.sqrt(Qx*Qx + Qy*Qy)

    return eval(calctype)

# These three functions are rotation matricies.  See, for example, http://en.wikipedia.org/wiki/Rotation_matrix#The_3-dimensional_rotation_matrices
def _rotateX(Y, Z, angle):
    """Rotate distance matricies about the X axis.
    """
    newY = Y*np.cos(angle) + Z*np.sin(angle)
    newZ = -Y*np.sin(angle) + Z*np.cos(angle)
    return newY, newZ

def _rotateY(X, Z, angle):
    """Rotate distance matricies about the Y axis.
    """
    newX = X*np.cos(angle) - Z*np.sin(angle)
    newZ = X*np.sin(angle) + Z*np.cos(angle)
    return newX, newZ

def energy_to_wavevector(energy):
    """Converts energy in eV to wavevector in Angstroms^{-1}.
    
    arguments:
        energy - energy, in eV
    returns:
        wavevector - in A^{-1}
    """
    assert energy > 0, "energy less than 0"
    # k=2*pi/lambda    
    return 2*np.pi/energy_to_wavelength(energy)

def energy_to_wavelength(energy):
    """Converts energy in eV to wavelength in Angstroms.

    arguments:
        energy - energy, in eV

    returns:
        wavelength - in A
    """
    assert energy > 0, "energy less than 0"
    # lam = hc/energy ; hc = 1239.84187433099714 eV*nm = 12398.4187433099714 eV*A
    return 12398.4187433099714/energy

def andor_ccd_efficiency(energy):
    """return the andor CCD DO436 detector efficiency (in percent) for a given
        energy.

    arguments:
        energy - energy at which the efficiency should be calculated

    returns:
        pct - detector efficiency in %
    """
    # energy vs quantum efficiency array, from DO436 manual. Traced with PlotDigitizer
    Energies = np.array([  1.23983,   1.30828,   1.55564, 1.67120,   1.78466,   1.92872, 2.07200,   2.52331,   2.75976, 3.14721,   3.30120,   3.32097, 3.44212,   3.65392,   3.78721, 3.97252,   4.31890,   4.39697, 4.61212,   5.01427,   5.32281, 5.65034,   5.92682,   6.32920, 10.0239,   20.0386,   30.0757, 100.479,   104.144,   301.476, 602.675,   742.764,   995.234, 1415.58,   1797.50,   1808.26, 3040.07,   4001.09,   5361.09, 9067.11,   10278.5,   20062.5])
    QEs = np.array([ 16.819, 30.713, 56.2157, 60.2377, 61.7916, 61.9744, 61.2431, 58.7751, 55.3016, 49.3601, 43.3272, 39.4881, 38.8483, 40.4936, 40.3108, 39.3053, 34.2779, 27.6051, 24.9543, 32.7239, 33.9123, 33.9123, 34.1865, 33.181, 18.0073, 20.3839, 25.0457, 44.6984, 25.3199, 39.9452, 65.0823, 85.192, 94.6984, 90.0366, 64.8995, 96.5265, 94.1499, 74.5887, 58.0439, 18.6472, 13.2541, 2.01097])

    assert type(energy) in (float, int), "energy must be float or int"
    assert Energies.min() < energy < Energies.max(), "energy must be between %f and %f." % (Energies.min(), Energies.max())

    return np.interp(energy, Energies, QEs)

def ccd_to_photons(img, energy, gain_readout=1e-6, guess_dark=True):
    """Convert a CCD image to number of photons.  This calculates the number of
        photons that would be incident assuming a detector of 100% efficiency.

    arguments:
        img - array to convert.  Can be 1d, 2d or 3d.
        energy - energy (in eV).
        gain_readout -readout time of the CCD.  Defaults to 1e-6 (1us).  This
            can be one of (1e-6, 2e-6, 16e-6, 32e-6).
        guess_dark - weather we should try and guess the dark count from the
            corners of the array.  If true, it subtracts this value off all of
            the pixels.  Defaults to True.

    returns:
        photon_array - array of photon counts that is the same size as the input
            array. The final array is casted as int and the values smaller than
            zero are removed.
    """
    assert isinstance(img, np.ndarray) and img.ndim in (1,2,3), "img must be an 1d, 2d, or 3d, array"
    assert type(energy) in (float, int), "energy must be float or int"
    # readtime and gain map from the do436 manual
    gain_readtime = {
        32e-6: 0.7,
        16e-6: 1.4,
        2e-6: 2,
        1e-6: 2
    }
    assert gain_readout in gain_readtime.keys(), "gain is not a recognized value"
    assert isinstance(guess_dark, bool)

    if guess_dark:
        if img.ndim == 3:
            avg = np.average(np.concatenate((img[:,0,0], img[:,-1,0], img[:,0,-1], img[:,-1,-1])))
            print "3d, avg", avg
        elif img.ndim == 2:
            avg = np.average(np.concatenate((img[0,0], img[-1,0], img[0,-1], img[-1,-1])))
            print "2d, avg", avg
        elif img.ndim == 1:
            avg = np.average(np.concatenate((img[0], img[-1])))
            print "1d, avg", avg

        img = img - avg

    gain = gain_readtime[gain_readout]
    QE = andor_ccd_efficiency(energy)
    PEconv = energy/3.65 # number of electrons created from one photon
    photonsPerCount = (100.0/QE) * gain/ PEconv
    img = img * photonsPerCount
    print "1 photon is %d counts" % int(1.0/photonsPerCount)
    return np.where(img.astype('int') < 0, 0, img.astype('int'))

