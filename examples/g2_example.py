import speckle

datafile = "data/xpcs/srt43-small.fits"
data = speckle.io.openfits(datafile)

# Draw a ROI in ds9. in DS9 select Region->Shape->Polygon and draw a region.
# Save the region with Region->Save Regions
mask = speckle.io.open_ds9_mask('./srt43-ROI.reg')

# or if you know the ROI as a 40x40 box centered at Row, Col (236, 118)
#mask = speckle.shape.rect(data[0].shape,40, 40, (236,118))

# Optionally, convert the movie to photons before calculating g2.
#data = speckle.scattering.ccd_to_photons(data, 778.1)

# calculate 100 frames of an XPCS movie (there are many other normalizations)
g2 = speckle.xpcs.g2_symm_norm(data, 100)

# optionally, save the g2 file once it's calculated so we can average later.
#speckle.io.writefits("srt43-g2.fits", g2)

# average g2 within the mask
avg_g2 = speckle.averaging.calculate_average(g2, mask)

# write to disk.
speckle.io.write_text_array("srt43-g2.txt", avg_g2)
