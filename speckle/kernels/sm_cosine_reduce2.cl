__kernel void execute(
    __global float2* cor_vals, // fft of correlation values
    __global float2* out_vals,
    __local  float2* buffer) // output buffer

{   
	
	int i = get_global_id(0); // this is the output pixel column
	int j = get_global_id(1); // this is the output pixel row
	int N = get_global_size(0); // this is the number of columns

	int global_col  = get_global_id(0);
	int global_row  = get_global_id(1);
	int local_col   = get_local_id(0);
	int local_row   = get_local_id(1);
	
	int global_cols = get_global_size(0);
	int local_cols  = get_local_size(0);
	
	int c  = (global_col+1)*2; // this is the component, and also the column in cor_vals
	int io = global_row*global_cols+global_col; // this is the index in out_vals
	int ic = global_row*512+c; // this is the index in cor_vals (angles = 512)
	
	int local_idx = local_row*global_cols+local_col;
	
	buffer[local_idx] = cor_vals[ic];
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	out_vals[io] = buffer[local_idx];
	
	//int c = (i+1)*2; // this is the component, and also the column in cor_vals
	//int io = j*N+i; // this is the index in out_vals
	//int ic = j*512+c; // this is the index in cor_vals (angles = 512)
	
	//float2 cv = cor_vals[ic];
	//float2 m = cv;//native_sqrt(cv.x*cv.x+cv.y*cv.y);
	
	//out_vals[io] = m/512.;

}