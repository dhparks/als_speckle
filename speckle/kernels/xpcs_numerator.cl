__kernel void execute(__global float* data,   // input data (a cube, frames by rows by columns)
		      __global float* ps_out, // output for numerator function (a cube, taus by rows by columns)
		      int tau,                // the value of tau for the current calculation
		      int depth)              // the number of frames in the input data
				  
{	

	int i    = get_global_id(0);
	int j    = get_global_id(1);
	int cols = get_global_size(0);
	int rows = get_global_size(1);
	
	float ps = 0;
	
	int index_out = i*cols+j;
	int index_in1 = 0;
	int index_in2 = 0;
	
	// now loop over all pairs of frames separated by tau. multiply
	// them together and stick the average result in ps_out.
	for (int k = 0; k < depth-tau; k++) {
		index_in1 = k*rows*cols+i*cols+j;
		index_in2 = (k+tau)*rows*cols+i*cols+j;
		
		p  = data[index_in1]*data[index_in2];
		ps = ps+p;
	}
	ps_out[index_out] = ps/k;
}





