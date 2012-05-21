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