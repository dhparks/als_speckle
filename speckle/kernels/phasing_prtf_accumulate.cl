__kernel void execute(__global float2 *fft,  // current fft
                      __global float2 *prtf) // accumulated prtf
    {
	
	// copy frame n out of scr and put it in dst. dst has size (rows,cols)
	// source has size (NxN) -- this should be changed in the future!
	
        int i = get_global_id(0);   // x coordinate
        int j = get_global_id(1);   // y coordinate
	int r = get_global_size(0); // rows
	int k = j*r+i;              // 1d coordinate from (i,j)
	
	// make phase factor
	float2 v = fft[k];
	float  h = hypot(v.x,v.y);
	
	// increment current accumulation
	float2 z = prtf[k];
	prtf[k] = z+(float2)(v.x/h,v.y/h);
    }