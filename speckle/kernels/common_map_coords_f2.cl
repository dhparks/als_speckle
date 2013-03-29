__kernel void execute( __global float2* input, int width_in, int height_in,
                       __global float2* output,
                       __global float* plan_x, __global float* plan_y, int interpolation_order)
        
        {

	int x = get_global_id(0);
	int y = get_global_id(1);
	int width_out  = get_global_size(0);
	int height_out = get_global_size(1);
	
	int i = x+width_out*y;
	float mapped_x = plan_x[i];
	float mapped_y = plan_y[i];
	 
	// nearest-neighbor interpolation. no "interpolation" per-se, just get the nearest pixel
	if (interpolation_order == 0) {
	    int j = round(mapped_x)+round(mapped_y)*width_in;
	    output[i] = input[j];
	 }
	 
	// bilinear interpolation (sample 4 closest neighbors)

        if (interpolation_order == 1) {
            int base_x = floor(mapped_x);
            int base_y = floor(mapped_y);

		// these are the four coordinates used to interpolate
		int x1 = base_x    ;
		int x2 = base_x + 1;
		int y1 = base_y    ;
		int y2 = base_y + 1;
		
		// enforce cyclic boundary conditions; coordinates can be negative
		int nx1 = x1;
		int nx2 = x2;
		int ny1 = y1;
		int ny2 = y2;
		if (x1 < 0 || x1 >= width_in)  {nx1 = (x1+width_in)%width_in;  }
		if (x2 < 0 || x2 >= width_in)  {nx2 = (x2+width_in)%width_in;  }
		if (y1 < 0 || y1 >= height_in) {ny1 = (y1+height_in)%height_in;}
		if (y2 < 0 || y2 >= height_in) {ny2 = (y2+height_in)%height_in;}

		// these are the distances from the neighbors to the interpolated point. use xi instead of nxi to preserve correct distance
		float dx1 = fabs(x1-mapped_x);
		float dx2 = fabs(x2-mapped_x);
		float dy1 = fabs(y1-mapped_y);
		float dy2 = fabs(y2-mapped_y);
		
		// the weights are a simple conversion of the distance
		float wx1 = 1.0f-dx1;
		float wx2 = 1.0f-dx2;
		float wy1 = 1.0f-dy1;
		float wy2 = 1.0f-dy2;
		
		// these are the values of the points at (x_i, y_i)
		float2 val11 = input[nx1+ny1*width_in];
		float2 val12 = input[nx1+ny2*width_in];
		float2 val21 = input[nx2+ny1*width_in];
		float2 val22 = input[nx2+ny2*width_in];
		
		// adding the pixel values and their weights is simple for the bilinear algorithm
		float re = val11.x*wx1*wy1+val12.x*wx1*wy2+val21.x*wx2*wy1+val22.x*wx2*wy2;
		float im = val11.y*wx1*wy1+val12.y*wx1*wy2+val21.y*wx2*wy1+val22.y*wx2*wy2;
		output[i] = (float2)(re,im);
		}

	
	}
