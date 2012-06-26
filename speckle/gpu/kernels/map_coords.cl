__kernel void execute( __global float* input, int width_in, int height_in,
                       __global float* output, int width_out, int height_out,
                       __global float* plan_x, __global float* plan_y, int interpolation_order)
        
        {

	int2 coords = (int2) (get_global_id(0), get_global_id(1)); // where to get coordinates from plan
	int i = coords.x+width_out*coords.y;
	 
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
		
		// enforce cyclic boundary conditions
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
		float val11 = input[nx1+ny1*width_in];
		float val12 = input[nx1+ny2*width_in];
		float val21 = input[nx2+ny1*width_in];
		float val22 = input[nx2+ny2*width_in];
		
		// adding the pixel values and their weights is simple for the bilinear algorithm
		float out_val = val11*wx1*wy1+val12*wx1*wy2+val21*wx2*wy1+val22*wx2*wy2;
		
		output[i] = out_val;}

	 // bicubic interpolation, very messy! could be made better with templating. loop are explicity unrolled so this should be pretty fast
	 if (interpolation_order == 3) {

		int base_y = floor(mapped_y);
		int base_x = floor(mapped_x);
		
		// these are the coordinates used to interpolate
		int x1 = base_x - 1;
		int x2 = base_x    ;
		int x3 = base_x + 1;
		int x4 = base_x + 2;
		int y1 = base_y - 1;
		int y2 = base_y    ;
		int y3 = base_y + 1;
		int y4 = base_y + 2;
		
		// enforce cyclic boundary conditions
		int nx1 = x1;
		int nx2 = x2;
		int nx3 = x3;
		int nx4 = x4;
		int ny1 = y1;
		int ny2 = y2;
		int ny3 = y3;
		int ny4 = y4;
		if (x1 < 0 || x1 >= width_in)  {nx1 = (x1+width_in)%width_in;  }
		if (x2 < 0 || x2 >= width_in)  {nx2 = (x2+width_in)%width_in;  }
		if (x3 < 0 || x3 >= width_in)  {nx3 = (x3+width_in)%width_in;  }
		if (x4 < 0 || x4 >= width_in)  {nx4 = (x4+width_in)%width_in;  }
		if (y1 < 0 || y1 >= height_in) {ny1 = (y1+height_in)%height_in;}
		if (y2 < 0 || y2 >= height_in) {ny2 = (y2+height_in)%height_in;}
		if (y3 < 0 || y3 >= height_in) {ny3 = (y3+height_in)%height_in;}
		if (y4 < 0 || y4 >= height_in) {ny4 = (y4+height_in)%height_in;}
	
		// distances to all the neighbor coordinates.
		float dx1 = fabs(x1-mapped_x);
		float dx2 = fabs(x2-mapped_x);
		float dx3 = fabs(x3-mapped_x);
		float dx4 = fabs(x4-mapped_x);
		float dy1 = fabs(y1-mapped_y);
		float dy2 = fabs(y2-mapped_y);
		float dy3 = fabs(y3-mapped_y);
		float dy4 = fabs(y4-mapped_y);
	
		// bicubic interpolation uses a piecewise-defined cubic function to set the weights.
		// the next section determines which part of the function to use and sets the weights.
		float weight_x1 = 0.0f;
		float weight_x2 = 0.0f;
		float weight_x3 = 0.0f;
		float weight_x4 = 0.0f;
		
		float weight_y1 = 0.0f;
		float weight_y2 = 0.0f;
		float weight_y3 = 0.0f;
		float weight_y4 = 0.0f;
	
		if (isless(dx1,1.0f)) { weight_x1 = 1.5*native_powr(dx1,3)-2.5*native_powr(dx1,2)+1; }
		if (isless(dx2,1.0f)) { weight_x2 = 1.5*native_powr(dx2,3)-2.5*native_powr(dx2,2)+1; }
		if (isless(dx3,1.0f)) { weight_x3 = 1.5*native_powr(dx3,3)-2.5*native_powr(dx3,2)+1; }
		if (isless(dx4,1.0f)) { weight_x4 = 1.5*native_powr(dx4,3)-2.5*native_powr(dx4,2)+1; }
	
		if (isless(dy1,1.0f)) { weight_y1 = 1.5*native_powr(dx1,3)-2.5*native_powr(dx1,2)+1; }
		if (isless(dy2,1.0f)) { weight_y2 = 1.5*native_powr(dy2,3)-2.5*native_powr(dy2,2)+1; }
		if (isless(dy3,1.0f)) { weight_y3 = 1.5*native_powr(dy3,3)-2.5*native_powr(dy3,2)+1; }
		if (isless(dy4,1.0f)) { weight_y4 = 1.5*native_powr(dy4,3)-2.5*native_powr(dy4,2)+1; }

		if (isgreaterequal(dx1,1.0f)*isless(dx1,2.0f)) { weight_x1 = -0.5*native_powr(dx1,3)+2.5*pown(dx1,2)-4*dx1+2; }
		if (isgreaterequal(dx2,1.0f)*isless(dx2,2.0f)) { weight_x2 = -0.5*native_powr(dx2,3)+2.5*pown(dx2,2)-4*dx2+2; }
		if (isgreaterequal(dx3,1.0f)*isless(dx3,2.0f)) { weight_x3 = -0.5*native_powr(dx3,3)+2.5*pown(dx3,2)-4*dx3+2; }
		if (isgreaterequal(dx4,1.0f)*isless(dx4,2.0f)) { weight_x4 = -0.5*native_powr(dx4,3)+2.5*pown(dx4,2)-4*dx4+2; }
	
		if (isgreaterequal(dy1,1.0f)*isless(dy1,2.0f)) { weight_y1 = -0.5*native_powr(dy1,3)+2.5*native_powr(dy1,2)-4*dy1+2; }
		if (isgreaterequal(dy2,1.0f)*isless(dy2,2.0f)) { weight_y2 = -0.5*native_powr(dy2,3)+2.5*native_powr(dy2,2)-4*dy2+2; }
		if (isgreaterequal(dy3,1.0f)*isless(dy3,2.0f)) { weight_y3 = -0.5*native_powr(dy3,3)+2.5*native_powr(dy3,2)-4*dy3+2; }
		if (isgreaterequal(dy4,1.0f)*isless(dy4,2.0f)) { weight_y4 = -0.5*native_powr(dy4,3)+2.5*native_powr(dy4,2)-4*dy4+2; }
		
		// get the values of the 16 neighbors
		float val11 = input[nx1+ny1*width_in];
		float val12 = input[nx1+ny2*width_in];
		float val13 = input[nx1+ny3*width_in];
		float val14 = input[nx1+ny4*width_in];
		
		float val21 = input[nx2+ny1*width_in];
		float val22 = input[nx2+ny2*width_in];
		float val23 = input[nx2+ny3*width_in];
		float val24 = input[nx2+ny4*width_in];
		
		float val31 = input[nx3+ny1*width_in];
		float val32 = input[nx3+ny2*width_in];
		float val33 = input[nx3+ny3*width_in];
		float val34 = input[nx3+ny4*width_in];
		
		float val41 = input[nx4+ny1*width_in];
		float val42 = input[nx4+ny2*width_in];
		float val43 = input[nx4+ny3*width_in];
		float val44 = input[nx4+ny4*width_in];
		
		// add the neighbor values with the correct weightings
		float out_val = val11*weight_x1*weight_y1+val12*weight_x1*weight_y2+val13*weight_x1*weight_y3+val14*weight_x1*weight_y4+
						val21*weight_x2*weight_y1+val22*weight_x2*weight_y2+val23*weight_x2*weight_y3+val24*weight_x2*weight_y4+
						val31*weight_x3*weight_y1+val32*weight_x3*weight_y2+val33*weight_x3*weight_y3+val34*weight_x3*weight_y4+
						val41*weight_x4*weight_y1+val42*weight_x4*weight_y2+val43*weight_x4*weight_y3+val44*weight_x4*weight_y4;
						
		output[i] = out_val;
		
	}
}"""