__kernel void execute(int frames,
		    __global float *goal,         // goal modulus
		    __global float *accumulation, // running rftf sum at each pixel
                    __global float2 *fourier)     // fourier modulus of current estimate
    {
	
	// recast (i,j) coords into 1d pixel index
	int i = get_global_id(0);   // x coordinate
	int j = get_global_id(1);   // y coordinate
	int r = get_global_size(0); // rows
	int k = j*r+i;              // 1d coordinate from (i,j)
	
	// for each pixel stack k, accumulate the error over
	// the n frames
	float g = goal[k];
	float cumulant = 0;
	for (int n=0; n < frames; n++) {
	    float d1 = g-fourier[k+n*r*r].x;
	    float d  = (d1*d1)/(g*g);
	    float e  = native_recip(native_sqrt(1+d));
	    cumulant = cumulant+e/frames;//fourier[k+n*r*r].x;
	}
		
	accumulation[k] = cumulant;
    }