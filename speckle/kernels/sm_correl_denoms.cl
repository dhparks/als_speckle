__kernel void execute(
    __global float2* in,
    __global float* out,
    __local float* buff) 
{   

    int angles = 512;
    int global_row = get_global_id(0);
    int local_row  = get_local_id(0);
    int row_offset = angles*global_row;

    float temp;
    float2 current;
    for (int k = 0; k < angles; k++) {
	current = in[k+row_offset];
	temp += native_sqrt(current.x*current.x+current.y*current.y);
    buff[local_row] = temp/angles;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    out[global_row] = buff[local_row];
	
}