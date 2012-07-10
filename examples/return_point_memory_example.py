import speckle

# Set up a base filename.  All of the files have the same format but a different number at tne end.
f = "/Volumes/Phaser/Work/Data/RawData/RPM/CoPd6201a/20090818/300K-pinhole0070-%03d.fits"

# these files are a subset of the 10 A (640 Oe) files that were used for the J. Appl. Phys. 
darkA = speckle.io.openfits(f % 55) # -80 A
imgA = speckle.io.openfits(f % 56) # 10 A
qacfA = speckle.io.openfits(f % 57) # 10 A
qacfdarkA = speckle.io.openfits(f % 58) # +80 A
darkB = speckle.io.openfits(f % 61) # -80 A
imgB = speckle.io.openfits(f % 62) # 10 A
qacfB = speckle.io.openfits(f % 63) # 10 A
qacfdarkB = speckle.io.openfits(f % 64) # +80 A

## Load the ROI from a ds9 region file
#ROI = speckle.io.open_ds9_mask('RPM-box-R215px.reg')
#ROI = speckle.io.open_ds9_mask('../300K-bgsub-pinhole0070-002.reg')

# or, alternatively, make your own using shape. This is a box of size l_x=146, l_y=124 centered at (x=1419, y=968). 
ROI = speckle.shape.rect(imgA.shape, (124,146), center=(968, 1419))

# call rot_memory()
result, intermediates = speckle.crosscorr.point_memory(imgA, imgB,
                                            darks=(darkA, darkB),
                                            qacfdarks = (qacfdarkA, qacfdarkB),
                                            qacfs=(qacfA, qacfB),
                                            mask = ROI,
                                            intermediates=True,
                                            flatten_width=30)

# Write out the intermediate arrays to the current directory
for file, data in intermediates.iteritems():
    speckle.io.writefits("%s.fits" % file, abs(data), overwrite=True)

print "Memory in this region is %f." % result
