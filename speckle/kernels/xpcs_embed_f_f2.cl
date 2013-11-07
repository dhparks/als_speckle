__kernel void execute(__global float* data_in,   // input data (rows to correlate along column axis)
		      __global float2* embedded, // output (data casted to cpx, zero-padded)
		      int L)                     // the number of columns in embedded
				  
{	

	// get the position of the current worker
	int row  = get_global_id(0);
	int col  = get_global_id(1);
	int cols = get_global_size(1);
	int idx1 = row*cols+col;
	
	// get the idx of the output
	int idx2 = row*L+col;
	
	// move the data to embedded
	embedded[idx2] = (float2)(data_in[idx1],0);
}





