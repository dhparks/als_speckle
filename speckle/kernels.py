slice_view = """
__kernel void execute(
    __global float* input,
    __global float* output,
    int N_in, int N_out, 
    int row, int col)  // row and col are start coords, N is output array size

// take a sub array from the master domains image

{
	// i_out and j_out are the x and y coordinate of the output image
	int i_out = get_global_id(0);
	int j_out = get_global_id(1);
	
	// i_in and j_in are the x and y coordinates of the input image
	int i_in = i_out+col;
	int j_in = j_out+row;
	
	// check to see if either is out of bounds; if it is, enforce cyclic boundary conditions
	if (i_in >= N_in || i_in < 0) {i_in = (i_in+N_in)%N_in;}
	if (j_in >= N_in || j_in < 0) {j_in = (j_in+N_in)%N_in;}
	
	output[i_out+N_out*j_out] = input[i_in+N_in*j_in];
}"""

median_filter3 = """
__kernel void execute(
    __global float* image, // image data
    float threshold)       // data/medfiltered_data > threshold is considered "hot")   

// take a sub array from the master domains image

{
    // i and j are the center coordinates
    int i = get_global_id(0);
    int j = get_global_id(1);
    int rows = get_global_size(0);
    int cols = get_global_size(1);
    
    float swap_min = 0.0f;
    float swap_max = 0.0f;
    float median = 0.0f;

    // pull the elements
    int x1 = j-1;
    int x2 = j;
    int x3 = j+1;
    
    int y1 = i-1;
    int y2 = i;
    int y3 = i+1;
    
    int nx1 = x1;
    int nx2 = x2;
    int nx3 = x3;
    
    int ny1 = y1;
    int ny2 = y2;
    int ny3 = y3;
    
    if (x1 < 0 || x1 >= cols) {nx1 = (x1+cols)%cols;}
    if (x2 < 0 || x2 >= cols) {nx2 = (x2+cols)%cols;}
    if (x3 < 0 || x3 >= cols) {nx3 = (x3+cols)%cols;}
    if (y1 < 0 || y1 >= rows) {ny1 = (y1+rows)%rows;}
    if (y2 < 0 || y2 >= rows) {ny2 = (y2+rows)%rows;}
    if (y3 < 0 || y3 >= rows) {ny3 = (y3+rows)%rows;}
    
    float r0 = image[nx1+ny1*cols];
    float r1 = image[nx2+ny1*cols];
    float r2 = image[nx3+ny1*cols];
    float r3 = image[nx1+ny2*cols];
    float r4 = image[nx2+ny2*cols];
    float r5 = image[nx3+ny2*cols];
    float r6 = image[nx1+ny3*cols];
    float r7 = image[nx2+ny3*cols];
    float r8 = image[nx3+ny3*cols];
    
    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r3);
    swap_max = fmax(r0,r3);
    r0 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r1,r2);
    swap_max = fmax(r1,r2);
    r1 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r7);
    swap_max = fmax(r4,r7);
    r4 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r5,r6);
    swap_max = fmax(r5,r6);
    r5 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r0,r7);
    swap_max = fmax(r0,r7);
    r0 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r1,r6);
    swap_max = fmax(r1,r6);
    r1 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r2,r5);
    swap_max = fmax(r2,r5);
    r2 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r3,r4);
    swap_max = fmax(r3,r4);
    r3 = swap_min;
    r4 = swap_max;

    swap_min = fmin(r0,r2);
    swap_max = fmax(r0,r2);
    r0 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r1,r3);
    swap_max = fmax(r1,r3);
    r1 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r5,r7);
    swap_max = fmax(r5,r7);
    r5 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r6);
    swap_max = fmax(r4,r6);
    r4 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r7,r8);
    swap_max = fmax(r7,r8);
    r7 = swap_min;
    r8 = swap_max;

    swap_min = fmin(r0,r4);
    swap_max = fmax(r0,r4);
    r0 = swap_min;
    r4 = swap_max;

    swap_min = fmin(r1,r5);
    swap_max = fmax(r1,r5);
    r1 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r2,r6);
    swap_max = fmax(r2,r6);
    r2 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r3,r7);
    swap_max = fmax(r3,r7);
    r3 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r0,r2);
    swap_max = fmax(r0,r2);
    r0 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r1,r3);
    swap_max = fmax(r1,r3);
    r1 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r4,r6);
    swap_max = fmax(r4,r6);
    r4 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r5,r7);
    swap_max = fmax(r5,r7);
    r5 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;
    
    median = r4;

    float q = image[j+rows*i]/median;
    if (q >= threshold) {image[j+rows*i] = median;}
    //image[j+rows*i] = r4;
}"""

magnet_transmit = """
__kernel void execute(
    float scale,
    float polarization,
    float2 n0,
    float2 dn,
    __global float* domains,
    __global float2* out)
{
	int i = get_global_id(0);
	
	float b = native_exp(-1*scale*(n0.y+polarization*domains[i]*dn.y));
	float d = scale*(n0.x+polarization*domains[i]*dn.x);
	
	out[i] = (float2)(b*native_cos(d),native_exp(b)*native_sin(d));

}"""


map_coords_buffer = """
__kernel void execute(
    __global float* input,
    int width_in,
    int height_in,
    __global float* output,
    int width_out,
    int height_out,
    __global float* plan_x,
    __global float* plan_y,
    int interpolation_order)
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

correl_denoms = """
__kernel void execute(
    __global float* in,
    __global float* out,
    int rows) 
{   
	
	int angles = 512;
	
	int j = get_global_id(0);
	float current = 0.0f;
	
	for (int k = 0; k < angles; k++) {
		current = in[k+j*angles];
		out[j] += current*current/rows;
	}
	
}"""

row_divide = """
__kernel void execute(
    __global float* corr,
    __global float* norm,
    __global float* out)
    
    // divide rows in an angular correlation by a normalization factor
{   
	
	int angles = 512;
	int i = get_global_id(0); // number of columns (512)
	int j = get_global_id(1); // number of rows
	float r = native_recip(norm[j]);
	out[i+j*angles] = corr[i+j*angles]*r;

}"""

cosine_reduce = """
__kernel void execute(
    __global float* cor_vals, // correlation values
    __global float* cos_vals, // cosine values
    __global float* out_vals, // output buffer
    int cos_rows              // number of rows in the cosine array
)
{   
	
	int i = get_global_id(0);
	int j = get_global_id(1); 
	
	for (int k = 0; k < 360; k++) {
		out_vals[i+cos_rows*j] += cor_vals[k+360*j]*cos_vals[k+360*i]/360;
	}
	
}"""

get_m0 = """
__kernel void execute(
    __global float* dft_re,
    __global float* dft_im,
    __global float2* m0) 					  
{m0[0] = (float2)(dft_re[0],dft_im[0]);}"""

replace_dc_component1 = """
__kernel void execute(
    __global float* input,
    __global float* output,
    int N)

{	
	int i = 0;
	int j = 0;

	// grab the 8 nearest neighbors of point (i,j). use modulo arithmetic to enforce cyclic boundary conditions.
	int nx1 = i-1;
	int nx2 = i;
	int nx3 = i+1;
	int ny1 = j-1;
	int ny2 = j;
	int ny3 = j+1;
	
	if (nx1 < 0 || nx1 >= N) {nx1 = (nx1+N)%N;}
	if (nx3 < 0 || nx1 >= N) {nx3 = (nx3+N)%N;}
	if (ny1 < 0 || ny1 >= N) {ny1 = (ny1+N)%N;}
	if (ny3 < 0 || ny3 >= N) {ny3 = (ny3+N)%N;}
	
	float val11 = input[nx1+ny1*N]; 
	float val12 = input[nx1+ny2*N];
	float val13 = input[nx1+ny3*N];
	float val21 = input[nx2+ny1*N];
	float val22 = input[nx2+ny2*N]; // this is the candidate value; the others are the neighbors
	float val23 = input[nx2+ny3*N];
	float val31 = input[nx3+ny1*N];
	float val32 = input[nx3+ny2*N];
	float val33 = input[nx3+ny3*N]; 
	
	// the function "compare" returns 0 if both inputs are the same sign and 1 if the signs differ. this is how a domain wall is detected.
	int out = (val11+val12+val13+val21+val23+val31+val32+val33)/8;
	output[0] = out;
}"""

replace_dc_component2 = """
__kernel void execute(
    __global float* in_re,
    __global float* in_im,
    __global float2* m0) // the components
{	
	in_re[0] = m0[0].x;
	in_im[0] = m0[0].y;
}"""

envelope_rescale = """
__kernel void execute(
    __global float* dft_re,
    __global float* dft_im,
    __global float* goal,
    __global float* blurred,
    __global float* rescaler) 					  
{	

	int i = get_global_id(0);
	float rescale = native_sqrt(goal[i]/blurred[i]); // sqrt because goal and blurred are intensity but dft_re and dft_im are field
	if (isnan(rescale)) {rescale = 0.0f;}

	dft_re[i] = dft_re[i]*rescale;
	dft_im[i] = dft_im[i]*rescale;
	rescaler[i] = rescale;
}"""

find_walls = """
__kernel void execute(
    __global float* input,
    __global float* walls,
    __global float* poswalls,
    __global float* negwalls,
    int N) // size

{	
	
	#define compare(a,b) (1-sign(a)*sign(b))/2
	
	// i and j are the x and y coordinates of the candidate pixel
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	// grab the 8 nearest neighbors of point (i,j). use modulo arithmetic to enforce cyclic boundary conditions.
	// loops have been manually unrolled for speed increase
	int nx1 = i-1;
	int nx2 = i;
	int nx3 = i+1;
	int ny1 = j-1;
	int ny2 = j;
	int ny3 = j+1;
	
	if (nx1 < 0 || nx1 >= N) {nx1 = (nx1+N)%N;}
	if (nx2 < 0 || nx2 >= N) {nx2 = (nx2+N)%N;}
	if (nx3 < 0 || nx3 >= N) {nx3 = (nx3+N)%N;}
	
	if (ny1 < 0 || ny1 >= N) {ny1 = (ny1+N)%N;}
	if (ny2 < 0 || ny2 >= N) {ny2 = (ny2+N)%N;}
	if (ny3 < 0 || ny3 >= N) {ny3 = (ny3+N)%N;}
	
	float val11 = input[nx1+ny1*N]; 
	float val12 = input[nx1+ny2*N];
	float val13 = input[nx1+ny3*N];
	float val21 = input[nx2+ny1*N];
	float val22 = input[nx2+ny2*N]; // this is the candidate value; the others are the neighbors
	float val23 = input[nx2+ny3*N];
	float val31 = input[nx3+ny1*N];
	float val32 = input[nx3+ny2*N];
	float val33 = input[nx3+ny3*N]; 
	
	// the function "compare" returns 0 if both inputs are the same sign and 1 if the signs differ. this is how a domain wall is detected.
	float preout = compare(val11,val22)+compare(val12,val22)+compare(val13,val22)+
				   compare(val21,val22)                     +compare(val23,val22)+ //skip comparing val22 to itself!
				   compare(val31,val22)+compare(val32,val22)+compare(val33,val22);
				 
	// write to the output buffers
	float out = 0.0f;
	float posout = 0.0f;
	float negout = 0.0f;
	
	if (preout > 0.1f) {out = 1.0f;}
	if (preout > 0.1f && val22 > 0.0f)  {posout = 1.0f;}
	if (preout > 0.1f && val22 <= 0.0f) {negout = 1.0f;}
	
	walls[i+N*j]    = out;
	poswalls[i+N*j] = posout;
	negwalls[i+N*j] = negout;
	
}"""

copy_to_buffer ="""
__kernel void execute(__global float2 *dst, __global float2 *src, int n, int L, int N)
        {
            int i_dst = get_global_id(0);
            int j_dst = get_global_id(1);
            int x0 = N/2-L/2;
            int y0 = N/2-L/2;
        
            // i_dst and j_dst are the coordinates of the destination. we "simply" need to turn them into 
            // the correct indices to move values from src to dst.
            
            int dst_index = (n*L*L)+(j_dst*L)+i_dst; // (frames)+(rows)+cols
            int src_index = (i_dst+x0)+(j_dst+y0)*N; // (cols)+(rows)
            
            dst[dst_index] = src[src_index];
        }"""




















