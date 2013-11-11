__kernel void execute(int frames,
		      __global float2 *fft,  // current fft
                      __global float *prtf) // accumulated prtf
    {
	
	// copy frame n out of scr and put it in dst. dst has size (rows,cols)
	// source has size (NxN) -- this should be changed in the future!
	
        int i = get_global_id(0);   // x coordinate
        int j = get_global_id(1);   // y coordinate
	int r = get_global_size(0); // rows
	int k = j*r+i;              // 1d coordinate from (i,j)
	
	// accumulate along the frame axis. compared to the earlier kernels
	// this reduces the number of memory requests by 2x.
	float2 cumulant = (float2)(0,0);
	for (int n=0; n<frames; n++) {
	    float2 v = fft[k+n*r*r]; // square arrays!
	    float  h = hypot(v.x,v.y);
	    cumulant = cumulant+(float2)(v.x/h,v.y/h);
	}

	// store the result in prtf
	cumulant = cumulant/frames;
	prtf[k]  = hypot(cumulant.x,cumulant.y);
	
	
	
    }