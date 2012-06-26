__kernel void execute(
        __global float* input, __global float* output,
        int N_in, int N_out,  int row, int col)  // row and col are start coords, N is output array size

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
}