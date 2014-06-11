__kernel void execute(
    __global float2* buffer,
    __global float* result,
    __local float* scratch) {

    int length = 512;
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    float2 b;
    
    // adapted from AMD code example
    
    // Load data into local memory
    if (global_index < length) {
	b = buffer[global_index];
	scratch[local_index] = native_sqrt(b.x*b.x+b.y*b.y);}
    else {scratch[local_index] = 0;}
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // parallel reduction
    for (int offset = get_local_size(0)/2; offset > 0; offset >>= 1) {
	if (local_index < offset) {
	    float other = scratch[local_index + offset];
	    float mine  = scratch[local_index];
	    scratch[local_index] = mine+other;}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // copy result to output
    if (local_index == 0) {result[get_group_id(0)] = scratch[0]/512.;}