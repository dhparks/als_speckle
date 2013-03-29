__kernel void execute(
        __global float2* input, __global float2* output,
        int cols_in, int rows_in,
        int row, int col)  // row and col are start coords, N is output array size

// take a sub array from the master domains image

{

	// i_out and j_out are the x and y coordinate of the output image
	int x_out = get_global_id(0);
	int y_out = get_global_id(1);
        
        int cols_out = get_global_size(0);
        int rows_out = get_global_size(1);
	
	// i_in and j_in are the x and y coordinates of the input image
	int x_in = x_out+col;
	int y_in = y_out+row;
	
	// check to see if either is out of bounds; if it is, enforce cyclic boundary conditions
	if (x_in >= cols_in || x_in < 0) {x_in = (x_in+rows_in)%rows_in;}
	if (y_in >= rows_in || y_in < 0) {y_in = (y_in+rows_in)%rows_in;}
	
	int out_index = x_out+cols_out*y_out;
	int in_index  = x_in+cols_in*y_in;
	
	output[out_index] = input[in_index];
	//output[out_index] = i_in;
}