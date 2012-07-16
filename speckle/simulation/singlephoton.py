import numpy as np

def sp_sim_xpcs_time(time, decaytime, scatterrate, clockperiod=40.0e-9):
    """Simulate a single photon XPCS signal with a decaytime over some time.
        The simulated sample has a scatterrate (in Hz) and detector has a
        clockperiod (in s).

    This algorithm implements the simulator from Rev. Sci. Instrum 74 4273.

    arguments:
        time - Total time to simulate.
        decaytime - programmed decay time, in seconds.
        scatterrate - The amount that the sample scatters, in Hz.
        clockperiod - Clock rate of the detector, in seconds. This is the time
            resolution of the experiment. Defaults to 40e-9 s (40 ns).

    returns:
        times - A (N x 1) array of photon incidence times. The final time * clockperiod ~= time.
    """
    import sys
    nevents = int(time/clockperiod)
    lam = clockperiod/decaytime
    
    sigmazsq = scatterrate/2.0
    num = (1.0 - np.exp(-2.0*lam) )
    den = (1.0 - np.exp(-1.0*lam) )**2
    sigmay = np.sqrt( sigmazsq * (num/den) )
    q2 = np.exp(-1.0*lam)
    q1 = (1.0 - q2 )

    Ex0, Ey0 = (0.0, 0.0)
    times = []
    i = 0
    while i < nevents:
        randvals = np.random.normal(size=2, scale=sigmay)

        Exi = q1*randvals[0] + q2*Ex0
        Eyi = q1*randvals[1] + q2*Ey0

        if np.random.poisson( clockperiod * ( Exi**2 + Eyi**2 ) ) >= 1:
            times.append( i*clockperiod )
            sys.stdout.write("\r%1.2es/%1.2fs (%03.3f%%)" % (i*clockperiod, time, i*100.0/float(nevents)))

        Ex0 = Exi
        Ey0 = Eyi

        i += 1
        sys.stdout.flush()

    print("")

    return np.array(times)

def sp_sim_xpcs_events(events, decaytime, scatterrate, clockperiod=40e-9):
    """Simulate a single photon XPCS signal with a decaytime for events events.
        The simulated sample has a scatterrate (in Hz) and detector has a
        clockperiod (in s).

    This algorithm implements the simulator from Rev. Sci. Instrum 74 4273.

    arguments:
        events - Total number of events to collect.
        decaytime - Decay time of simulated dataset, in seconds.
        scatterrate - The amount that the sample scatters, in Hz.
        clockperiod - Clock rate of the detector, in s.  This is the time
            resolution of the experiment. Defaults to 40e-9 s (40 ns).

    returns:
        times - A (events x 1) array of photon incidence times.
    """
    import sys
    lam = clockperiod/decaytime
    
    sigmazsq = scatterrate/2.0
    num = (1.0 - np.exp(-2.0*lam) )
    den = (1.0 - np.exp(-1.0*lam) )**2
    sigmay = np.sqrt( sigmazsq * (num/den) )
    q2 = np.exp(-1.0*lam)
    q1 = (1.0 - q2 )

    i = 0
    nevents = 0
    Ex0, Ey0 = (0.0, 0.0)
    # quantatively, atATime should scale like decaytime/scatterrate
    atATime = int(1e13*decaytime/scatterrate)
    if atATime < 500:
        atATime = 500
    elif atATime > 1e7:
        atATime = int(1e7)

    Exy = np.zeros((2,atATime))
    # list of times, add zero so when we check for deadtime the list isn't empty. Remove it at the end.
    incidencetimes = np.zeros(events, dtype='int64')

#    print("buffering %d events at a time\n" % atATime)
    while nevents < events:
        Exy = np.random.normal(size=(2, atATime), scale = sigmay) * q1
        Exy[0, 0] += Ex0*q2
        Exy[1, 0] += Ey0*q2

        for j in range(1,atATime):
            Exy[:,j] += Exy[:,j-1]*q2

        R = clockperiod * (Exy**2).sum(axis=0)
        eventsRemaining = events - nevents
        ridx = np.argwhere(np.random.poisson( R ) >= 1)[0:eventsRemaining]
        for t in ridx:
            incidencetimes[nevents] = i+t
            nevents += 1
            sys.stdout.write("\rfound event %d/%d @ %d. R = %f" % (nevents, events, i+t, R[t]))

        i += atATime
        [Ex0, Ey0] = Exy[:,-1]
        sys.stdout.flush()

    print("")

    return incidencetimes
