__kernel void execute(__global float* primary,
		      __global float* rescale,
		      __global float* weights,
		      __global float* out)
        
        {

	int kmax = 100;
	
	// get coordinates
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(0);
	int M = get_global_size(1);
	
	int x = (i+N/2)%N-N/2; // this coordinate system has its origin at center
	int y = (j+N/2)%N-N/2;
	float fx = (float)x; // coordinates as float
	float fy = (float)y;
	
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
	float iv = 0;
	
	int keep_x1 = 1;
	int keep_x2 = 1;
	int keep_y1 = 1;
	int keep_y2 = 1;

	// for each slice of the spectrum, do a bilinear interpolation to find
	// the pixel which contributes to the pixel (i,j) in the output image.
	for (int k = 0; k < kmax; k++) { //kmax is the number of samples in the spectrum
	    
	    rv = rescale[k]; // rescaling factor
	    sw = weights[k]; // spectral weight
		
		rx = fx/rv;
		ry = fy/rv;
		
	    //rx = fmod(rx+N,N); // rescale x, then modulo coords to corner-centered system
	    //ry = fmod(ry+M,M);
		
		keep_x1 = 1;
		keep_x2 = 1;
		keep_y1 = 1;
		keep_y2 = 1;
	    
	    // these form the coordinates which we will use to interpolate. they are still
		// in the (-N/2,N/2) coordinate system
	    x1 = floor(rx);
	    x2 = x1+1;
	    y1 = floor(ry);
	    y2 = y1+1;

		// check to see if the location we pull from is out-of-bounds
		if (x1 <= -N/2) {keep_x1 = 0;}
		if (x1 >= N/2)  {keep_x1 = 0;}
		if (x2 <= -N/2) {keep_x2 = 0;}
		if (x2 >= N/2)  {keep_x2 = 0;}
		if (y1 <= -N/2) {keep_y1 = 0;}
		if (y1 >= N/2)  {keep_y1 = 0;}
		if (y2 <= -N/2) {keep_y2 = 0;}
		if (y2 >= N/2)  {keep_y2 = 0;}
		
		// these are the weights for each coordinate
	    wx1 = (1-fabs(x1-rx))*keep_x1;
	    wx2 = (1-fabs(x2-rx))*keep_x2;
	    wy1 = (1-fabs(y1-ry))*keep_y1;
	    wy2 = (1-fabs(y2-ry))*keep_y2;
		
		// now modulo the coordinates to bring them back to the (0,N) system
		if (x1 < 0) {x1 = (x1+N)%N;}
		if (x2 < 0) {x2 = (x2+N)%N;}
		if (y1 < 0) {y1 = (y1+M)%M;}
		if (y2 < 0) {y2 = (y2+M)%M;}

		// cyclic boundary conditions
	    if (x2 >= N) {x2 = (x2+N)%N;}
	    if (y2 >= M) {y2 = (y2+M)%M;}
	    
	    // these are the values from the four pixels
	    val11 = primary[x1+y1*N];
	    val12 = primary[x1+y2*N];
	    val21 = primary[x2+y1*N];
	    val22 = primary[x2+y2*N];
	    
	    // this is the interpolated value at the fractional coordinate (rx,ry)
	    iv = val11*wx1*wy1+val12*wx1*wy2+val21*wx2*wy1+val22*wx2*wy2;
	    
	    // add to cumulant, ending the loop
	    cumulant += iv*sw;

	}

	out[j*N+i] = cumulant;
	
}