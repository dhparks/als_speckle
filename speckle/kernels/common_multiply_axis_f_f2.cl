__kernel void execute(int frames,
		    __global float  *in1,    // 2d float input
		    __global float2 *in2,    // 3d complex input
                    __global float2 *output) // 3d complex output
    {
	
	// recast (i,j) coords into 1d pixel index
	int i = get_global_id(0);   // x coordinate
	int j = get_global_id(1);   // y coordinate
	int r = get_global_size(0); // rows
	int k = j*r+i;              // 1d coordinate from (i,j)
	
	// for each pixel stack k, accumulate the error over
	// the n frames
	float a = in1[k];
	for (int n=0; n<frames; n++) {
	    float2 b = in2[k+n*r*r];
	    output[k+n*r*r] = (float2)(a*b.x,a*b.y);
	}
    }