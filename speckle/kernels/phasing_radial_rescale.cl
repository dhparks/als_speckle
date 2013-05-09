__kernel void execute(__global float2* primary, // primary comes in as float2 but has zero imaginary component.
		      __global const float* rescale,
		      __global const float* weights,
		      int kmax,
		      __global float2* out) // out is float2 but has zero imaginary component
        
        {
	// this kernel assumes that the data being passed is in human-centered form, meaning that
	// the center of the diffraction pattern is at coordinate (N2,M/2) instead of (0,0).
	// this arrangement is done on the gpu by invoking common_fftshift_f.cl or
	// common_fftshift_f2.cl depending on data type. this split is intended to facilitate easier
	// implementation for algorithms such as this one at the cost of some speed necessary to do
	// the shifting of the data between coordinates.

	// get coordinates of local worker
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(0);
	int M = get_global_size(1);
	
	int N2 = N/2;
	int M2 = M/2;
	
	// switch from machine coordinates (i, j) to human coordinates (x, y)
	float x = (float) i;
	float y = (float) j;
	if (i > N2-1) {x = i-N;}
	if (j > M2-1) {y = j-N;}
	
	float cumulant = 0; // this holds the running total of the value
	
	// these are used inside the loop
	float rv = 0;
	float sw = 0;
	float rx = 0;
	float ry = 0;
	int x1 = 0;
	int x2 = 0;
	int y1 = 0;
	int y2 = 0;
	float wx1 = 0;
	float wx2 = 0;
	float wy1 = 0;
	float wy2 = 0;
	float val11 = 0;
	float val12 = 0;
	float val21 = 0;
	float val22 = 0;
	int keep_x1 = 1;
	int keep_x2 = 1;
	int keep_y1 = 1;
	int keep_y2 = 1;

	float iv = 0;

	// for each slice of the spectrum, do a bilinear interpolation to find
	// the pixel which contributes to the pixel (i,j) in the output image.
	for (int k = 0; k < kmax; k++) { //kmax is the number of samples in the spectrum
	    
	    rv = rescale[k]; // rescaling factor
	    sw = weights[k]; // spectral weight
		
	    rx = x/rv;
	    ry = y/rv;

	    // these form the coordinates which we will use to interpolate. they are still
	    // in the (-N2,N2) coordinate system
	    x1 = floor(rx);
	    x2 = x1+1;
	    y1 = floor(ry);
	    y2 = y1+1;

	    // check to see if the location we pull from is out-of-bounds
	    keep_x1 = 1;
	    keep_x2 = 1;
	    keep_y1 = 1;
	    keep_y2 = 1;
	    if (x1 <= -N2) {keep_x1 = 0;}
	    if (x1 >= N2)  {keep_x1 = 0;}
	    if (x2 <= -N2) {keep_x2 = 0;}
	    if (x2 >= N2)  {keep_x2 = 0;}
	    if (y1 <= -M2) {keep_y1 = 0;}
	    if (y1 >= M2)  {keep_y1 = 0;}
	    if (y2 <= -M2) {keep_y2 = 0;}
	    if (y2 >= M2)  {keep_y2 = 0;}
		
	    // these are the weights for each coordinate
	    wx1 = (1-fabs(x1-rx))*keep_x1;
	    wx2 = (1-fabs(x2-rx))*keep_x2;
	    wy1 = (1-fabs(y1-ry))*keep_y1;
	    wy2 = (1-fabs(y2-ry))*keep_y2;
		
	    // now modulo the coordinates to bring them back to the (0,N) system
	    if (x1 < 0) {x1 = x1+N;}
	    if (x2 < 0) {x2 = x2+N;}
	    if (y1 < 0) {y1 = y1+M;}
	    if (y2 < 0) {y2 = y2+M;}

	    // these are the values from the four pixels
	    val11 = primary[x1+y1*N].x;
	    val12 = primary[x1+y2*N].x;
	    val21 = primary[x2+y1*N].x;
	    val22 = primary[x2+y2*N].x;
	    
	    // this is the interpolated value at the fractional coordinate (rx,ry)
	    iv = val11*wx1*wy1+val12*wx1*wy2+val21*wx2*wy1+val22*wx2*wy2;
	    
	    // add to cumulant, ending the loop
	    cumulant += iv*sw;

	}

	out[j*N+i] = (float2)(cumulant,0);
	
}