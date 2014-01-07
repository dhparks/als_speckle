__kernel void execute(__global float* data_in, int cols)   // input data to normalize
				  
{	

	// get the position of the current worker
	int row  = get_global_id(0);
	int idx  = 0;
        
        for (int k = 0; k < cols; k++) {
            idx = row*cols+k;
            data_in[idx] = data_in[idx]/(cols-k);

        }
        
}





