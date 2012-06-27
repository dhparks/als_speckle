import speckle
import numpy

pcfile = "4000V_Xray4.fits"

# open a photon counting fits file, correcting overflows and sorting the data by increasing incidence times
data = speckle.io.open_photon_counting_fits(pcfile, correct=True, sort=True)

# The single photon functions are in speckle.xpcs.sp_*

# bin data into frames of 1e-3 s each and 32 x 32.
# this can take a while and a large amount of ram
#binned_data = speckle.xpcs.sp_bin_by_space_and_time(data, 1e-3, 32)

# Sum all of the data into one image.
sum_all = speckle.xpcs.sp_sum_bin_all(data, xybin=1)

# write summed file
speckle.io.writefits("4000V_Xray4-summed.fits", sum_all)

# correlate the photons that are incident on x=[500:800] and y=[1400:1800]
corr, corrtime = speckle.xpcs.sp_autocorrelation_range(data, (500,800), (1400,1800))

corr_combined = numpy.column_stack((corrtime, corr))

# convert to clock time by multiplying by clock frequency
corr_combined[:, 0] = corr_combined[:, 0] * 40e-9

speckle.io.write_text_array("4000V_Xray4-cc.txt", corr_combined)

################################################################################
# simulate 10k events with decay time of 4.3e-6 s and scatter rate of 5e5 hz.
events = speckle.xpcs.sp_sim_xpcs_events(10000, 4.3e-6, 5e5)

# events is an 1d array of photon incidence times.  We can pass this directly to sp_autocorrelation().

# combine, convert, and write data
corr, corrtime = speckle.xpcs.sp_autocorrelation(events)
corr_combined = numpy.column_stack((corrtime, corr))
corr_combined[:, 0] = corr_combined[:, 0] * 40e-9
speckle.io.write_text_array("AC-4.3e-6s_decay.txt", corr_combined)
