#!/usr/bin/env python
import argparse
import numpy as np
import speckle

def align_match(img1, img2, ROI=False):
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

def align_match_row_merge(img1, img2, mergeROI, fitROI, regions=(0, 2048, 256), direction='rows', blend_width=10):
    """ Merge two images together by aligning and matching counts. This function
    differs from align_match_merge() in that there image is split
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
        blend_width - type of merge to do.  Can be 'hard' or 'soft'.  Soft merges
            feather the edges and average the two exposures at the edge.

    returns:
        img1 with img2 merged in.
    """
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
            merge_part = speckle.conditioning.merge(img1, alignImg2, maskFitROI, maskMergeROI, blend_width)
        else:
            merge_part = merge_part * (1 - maskMergeROI) + match_merge(img1, alignImg2, maskMergeROI, maskFitROI, blend_width) * maskMergeROI

        iteration += 1

    return merge_part

def align_match_merge(img1, img2, mergeROI, fitROI, blend_width=10):
    """ Align images, match counts, and merge the image.

    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        fitROI - Region of interest for match_counts
        blend_width - Width for the blending, in pixels. Defaults to 10 px. If
            blend_width = 0, it does a hard merge.

    returns:
        img1 with img2 merged in.
    """
    alignedImg2 = speckle.conditioning.align_frames(img2, align_to=img1, region=ROI)
    return speckle.conditioning.merge(img1, alignedImg2, fitROI, mergeROI, blend_width)

def match_merge(img1, img2, mergeROI, fitROI, blend_width=10):
    """ merge two images together by matching counts.

    arguments:
        img1 - The 'main' image.  img2 is merged into it.
        img2 - Image to merge in.
        mergeROI - Region from img1 that needs to be merged
        fitROI - Region of interest for MergeCounts
        blend_width - Width for the blending, in pixels. Defaults to 10 px. If
            blend_width = 0, it does a hard merge.

    returns:
        img1 with img2 merged in.
    """
    alignImg2 = speckle.conditioning.match_counts(img1, img2, region=fitROI)
    return speckle.conditioning.merge(img1, alignImg2, fitROI, mergeROI, blend_width)

def to2D(img):
    if img.ndim == 3:
        return img[0]
    else:
        return img

def createRegAndMerge(file1, file2, mergeROI, fitROI, blend_width):
    maskToReplace = to2D(speckle.io.open_ds9_mask(mergeROI))
    maskToFit = to2D(speckle.io.open_ds9_mask(fitROI))
    f1 = to2D(speckle.io.openfits(file1))
    f2 = to2D(speckle.io.openfits(file2))
    return align_match_merge(f1, f2, maskToReplace, maskToFit, blend_width)

def createRegAndMergeRows(file1, file2, mergeROI, fitROI, rows, direction, width):
    return align_match_row_merge(to2D(speckle.io.openfits(file1)), 
                                            to2D(speckle.io.openfits(file2)),
                                            to2D(speckle.io.open_ds9_mask(mergeROI)),
                                            to2D(speckle.io.open_ds9_mask(fitROI)),
                                            rows,
                                            direction,
                                            width)  

parser = argparse.ArgumentParser(description='Merge two FITS files together over regions of interest.  This will match the counts in FitRegion and merge file_from2 into file_into1 with MergeRegion. The merge can also be done region-by-region by specificing -rs with a direction (either -r or -c for rows and cols, respectively).')

parser.add_argument('file_into1', help='FITS file to merge into.')
parser.add_argument('file_from2', help='FITS file to merge from')
parser.add_argument('FitRegion', help='DS9 region file to fit')
parser.add_argument('MergeRegion', help='DS9 region file to merge')
parser.add_argument('-w', '--width', nargs=1, type=int, help='width for blending the merge, in pixels')
parser.add_argument('-rs', '--regionstep', nargs=3, type=int, help='Specify how the data is split up when fitting regions. Should be three values of (region_start, region_stop, region_step).')
parser.add_argument('-r', '--row', action="store_true", help='Separate regions by rows')
parser.add_argument('-c', '--column', action="store_true", help='Separate regions by columns')

args = parser.parse_args()

if isinstance(args.width, (tuple, list)):
    args.width = args.width[0]
if args.row:
    direction = 'rows'
elif args.column:
    direction = 'cols'
else:
    direction = False

if not direction and args.regionstep:
    direction = 'rows'
    print("no direction for regions given, using %s." % direction)

print args
outf = args.file_into1.split("/")[-1].replace(".fits", "") + "_replaced_" + args.file_from2.split("/")[-1].replace(".fits", "") + ".fits"

if direction:   
    speckle.io.writefits(outf, createRegAndMergeRows(args.file_into1, args.file_from2, args.MergeRegion, args.FitRegion, args.regionstep, direction, args.width), overwrite=True)
else:
    speckle.io.writefits(outf, createRegAndMerge(args.file_into1, args.file_from2, args.MergeRegion, args.FitRegion, args.width), overwrite=True)


