#!/usr/bin/env python
import argparse
import numpy as np
import speckle

def distance_transform(mask):
    """ Implements a Euclidean distance transform from a binary mask.
    Implementation is from http://www.logarithmic.net/pfh/blog/01185880752
    arguments:
        mask - the binary mask.
    returns
        img - an array of shortest distance to the pixel from that binary mask.
    """

    def _upscan(f):
        """ A Helper function for distance_transform()
        """
        for i, fi in enumerate(f):
            if fi == np.inf: continue
            for j in xrange(1,i+1):
                x = fi+j*j
                if f[i-j] < x: break
                f[i-j] = x

    f = np.where(mask, 0.0, np.inf)
    for i in xrange(f.shape[0]):
        _upscan(f[i,:])
        _upscan(f[i,::-1])
    for i in xrange(f.shape[1]):
        _upscan(f[:,i])
        _upscan(f[::-1,i])
    np.sqrt(f,f)
    return f

def alignAndMatchCounts(img1, img2, ROI=False):
    """ Align two images and match the counts of the images.

    arguments:
        img1 - 1st image to align.
        img2 - 2nd image to align.
        ROI - binary masked region of interest where the alignment and matching of counts should occur.

    returns:
        img2 - img2 is returned after being aligned and counts matched.
    """
    alignedImg2 = speckle.conditioning.align_frames(img2, align_to=img1, region=ROI)
    return speckle.conditioning.match_counts(img1, alignedImg2, region=ROI)

def rowMergeImageByAlignAndMatchCounts(img1, img2, mergeROI, fitROI, regions=(0, 2048, 256), direction='rows', mergeType='hard'):
    """ Merge two images together by aligning and matching counts. This function
    differs from mergeImageByAlignAndMatchCounts() in that there image is split
    up into rows and merged a region at a time.  This is useful when trying to
    merge the data where the beam block wire is located.  In this area, it's
    advantageous to fit them in small regions.

    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        fitROI - Region of interest for MergeCounts
        regions - a region area (rmin, rmax, rstep) of how the regions should be
            fit.  The fitROI is applied first, then the regions are selected out
            and fitted manually.
        direction - directions
        mergeType - type of merge to do.  Can be 'hard' or 'soft'.  Soft merges
            feather the edges and average the two exposures at the edge.

    returns:
        img1 with img2 merged in.
    """
    """ XXX I Forgot what this did?"""

    alignImg2 = speckle.conditioning.align_frames(img2, align_to=img1, region=mergeROI)
    reg_start, reg_stop, reg_step = regions
    (ys, xs) = img1.shape
    if direction == 'rows':
        maxshape = ys
    else:
        maxshape = xs

    if (reg_start < 0) or (reg_start > reg_stop):
        reg_start = 0
    if reg_stop > maxshape:
        reg_stop == maxshape
    if reg_step > reg_stop - reg_start:
        reg_step = reg_stop - reg_start

    iteration = 0
    for rs in range(reg_start, reg_stop, reg_step):

        rowmask = np.zeros_like(mergeROI)
        if direction == 'rows':
            rowmask[rs:rs+reg_step, :] = 1
        else:
            rowmask[:, rs:rs+reg_step] = 1
        
        maskMergeROI = mergeROI * rowmask # this is Region to replace, masked
        maskFitROI = fitROI * rowmask # Region to Fit, masked

        if mergeROI.sum() == 0:
            iteration += 1
            continue

        if iteration == 0:
            merge_part = mergeImageByMatchCounts(img1, alignImg2, maskMergeROI, maskFitROI, mergeType)
        else:
            merge_part = merge_part * (1 - maskMergeROI) + mergeImageByMatchCounts(img1, alignImg2, maskMergeROI, maskFitROI, mergeType) * maskMergeROI

        iteration += 1

    return merge_part

def mergeImageByAlignAndMatchCounts(img1, img2, mergeROI, fitROI, mergeType='hard'):
    """ merge two images together by aligning and matching counts.

    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        fitROI - Region of interest for MergeCounts
        mergeType - type of merge to do.  Can be 'hard' or 'soft'.  Soft merges
            feather the edges and average the two exposures at the edge.

    returns:
        img1 with img2 merged in.
    """
    alignImg2 = alignAndMatchCounts(img1, img2, fitROI)
    return mergeImage(img1, alignImg2, mergeROI, mergeType)

def mergeImageByMatchCounts(img1, img2, mergeROI, fitROI, mergeType='hard'):
    """ merge two images together by  matching counts.

    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        fitROI - Region of interest for MergeCounts
        mergeType - type of merge to do.  Can be 'hard' or 'soft'.  Soft merges
            feather the edges and average the two exposures at the edge.

    returns:
        img1 with img2 merged in.
    """
    alignImg2 = speckle.conditioning.match_counts(img1, img2, region=fitROI)
    return mergeImage(img1, alignImg2, mergeROI, mergeType)

def mergeImage(img1, img2, mergeROI, mergeType):
    """ merge two images together.
    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        mergeType - type of merge to do.  Can be hard or soft.  Soft merges feather the edges and average the two exposures at the edge.
    returns:
        img1 with img2 merged in.
    """
    from scipy.special import erf
    if mergeType == 'hard':
        return np.where(mergeROI, img2, img1)
    else:
        # maxfromBB is the maximum distance from the BB that we replace
        maxfromBB = 12.

        # distancemap is an expensive calculation, so only calculate it in the
        # relevant distance around mergeROI
        bounds = bound(mergeROI,force_to_square=False,pad=int(2*maxfromBB))
        r0,c0 = bounds[0],bounds[2]
        rows = int(bounds[1]-bounds[0])
        cols = int(bounds[3]-bounds[2])
        subROI = mergeROI[r0:r0+rows,c0:c0+cols]
        distancemap = distance_transform(subROI)

        img1Mask = (erf( (distancemap-maxfromBB)/12 ) + 1. )/2. # factor inside erf() smooths it out a little
        blender = numpy.ones_like(img1)
        blender[r0:r0+rows,c0:c0+cols] = img1Mask
        
        return img1*blender + img2*(1-blender)

def to2D(img):
    if img.ndim == 3:
        return img[0]
    else:
        return img

def openAndMerge(file1, file2, fileMaskReplace, fileMaskFit, mergeType = 'hard'):
    return mergeImageByAlignAndMatchCounts(to2D(speckle.io.openfits(file1)), 
                                            to2D(speckle.io.openfits(file2)),
                                            to2D(speckle.io.openfits(fileMaskReplace)),
                                            to2D(speckle.io.openfits(fileMaskFit)),
                                            mergeType)

def createRegAndMerge(file1, file2, mergeROI, fitROI, mergeType):
    maskToReplace = speckle.io.open_ds9_mask(mergeROI)
    maskToFit = speckle.io.open_ds9_mask(fitROI)
    return mergeImageByAlignAndMatchCounts(to2D(speckle.io.openfits(file1)), 
                                            to2D(speckle.io.openfits(file2)),
                                            to2D(maskToReplace),
                                            to2D(maskToFit),
                                            mergeType)

def createRegAndMergeRows(file1, file2, mergeROI, fitROI, rows, direction, mergeType):
    maskToReplace = speckle.io.open_ds9_mask(mergeROI)
    maskToFit = speckle.io.open_ds9_mask(fitROI)
    return rowMergeImageByAlignAndMatchCounts(to2D(speckle.io.openfits(file1)), 
                                            to2D(speckle.io.openfits(file2)),
                                            to2D(maskToReplace),
                                            to2D(maskToFit),
                                            rows,
                                            direction,
                                            mergeType)  

parser = argparse.ArgumentParser(description='Merge two FITS files together over regions of interest.  This will match the counts in FitRegion and merge file_from2 into file_into1 with MergeRegion. The merge can also be done region-by-region by specificing -rs with a direction (either -r or -c for rows and cols, respectively).')

parser.add_argument('file_into1', help='FITS file to merge into.')
parser.add_argument('file_from2', help='FITS file to merge from')
parser.add_argument('FitRegion', help='DS9 region file to fit')
parser.add_argument('MergeRegion', help='DS9 region file to merge')
parser.add_argument('-t', '--type', nargs=1, help='Merge type. Can be either hard or soft', default='hard')
parser.add_argument('-rs', '--regionstep', nargs=3, type=int, help='Specify how the data is split up when fitting regions. Should be three values of (region_start, region_stop, region_step).')
parser.add_argument('-r', '--row', action="store_true", help='Separate regions by rows')
parser.add_argument('-c', '--column', action="store_true", help='Separate regions by columns')
#parser.add_argument('-o', '--out', nargs=1, help='outfile', default='mergedFile.fits')

args = parser.parse_args()

if args.type not in ('hard', 'soft'):
    args.type = 'hard'

if args.row:
    direction = 'rows'
elif args.column:
    direction = 'cols'
else:
    direction = False

if not direction and args.regionstep:
    direction = 'rows'
    print("no direction for regions given, using %s." % direction)

outf = args.file_into1.split("/")[-1].replace(".fits", "") + "_replaced_" + args.file_from2.split("/")[-1].replace(".fits", "") + ".fits"

if direction:   
    speckle.io.writefits(outf, createRegAndMergeRows(args.file_into1, args.file_from2, args.MergeRegion, args.FitRegion, args.regionstep, direction, args.type), overwrite=True)
else:
    speckle.io.writefits(outf, createRegAndMerge(args.file_into1, args.file_from2, args.MergeRegion, args.FitRegion, args.type), overwrite=True)


