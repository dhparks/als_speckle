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
(yc, xc) = (1140, 1299)

# call rot_memory()
result, intermediates = speckle.crosscorr.rot_memory(imgA, imgB, (yc,xc),
                                            darks=(darkA, darkB),
                                            qacfdarks = (qacfdarkA, qacfdarkB),
                                            qacfs=(qacfA, qacfB),
                                            Rvals = 25,
                                            intermediates=True,
                                            flatten_width=False)

# Write out the intermediate arrays to disk
for file, data in intermediates.iteritems():
    speckle.io.writefits("%s.fits" % file, abs(data), overwrite=True)

# write out the final results.  The result array has two columns, formatted as (radial distance, memory)
hdr_str = "# Rotational memory between RPM files (55, 62), with (%d, %d) as  the center.\n" % (xc, yc)
speckle.io.write_text_array("RPM-qdep-memory.txt", result, header=hdr_str)
