__kernel void execute(__global float* data,  // input data (a cube, frames by rows by columns)
		      __global float* s_out, // output for numerator function (a cube, taus by rows by columns)
		      int cols, int rows)    // the number of frames in the input data
				  
	// given the 3d input, calculate the average along the 2 spatial axes. the result is 1d.
				  
{	

	int frame = get_global_id(0);
	int depth = get_global_size(0);
	
	float s = 0;

	for (int row = 0; row < rows; row++){
		for (int col = 0; col < cols; col++){
			x = frame*rows*cols+row*cols+col;
			s = s+data[x];
		}
	}
	
	s_out[frame] = s;

}





