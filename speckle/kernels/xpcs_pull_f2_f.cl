__kernel void execute(__global float2* cpx,   // input data (rows to correlate along column axis)
		      __global float2* flt, // output (data casted to cpx, zero-padded)
		      int L)                     // the number of columns in embedded
				  
{	

	// get the position of the current worker in cpx
	int row  = get_global_id(0);
	int col  = get_global_id(1);
	int idx1 = row*L+col;
	
	// get the idx of the output int flt
	int cols = get_global_size(1);
	int idx2 = row*cols+col;
	
	// move the data to embedded
	flt[idx2] = data_in[idx1].x;
}





