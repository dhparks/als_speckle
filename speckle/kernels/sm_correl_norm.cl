__kernel void execute(
    __global float2* in,     
    __global float* denoms,
    __local float* locald,
    __global float2* out)

{   

    int angles = 512;
    int i = get_global_id(0); // col
    int j = get_global_id(1); // row
    
    int global_col = get_global_id(0);
    int global_row = get_global_id(1);
    int local_col  = get_local_id(0);
    int local_row  = get_local_id(1);

    // copy denoms to local
    if (local_col == 0) {
	float d = denoms[global_row];
	locald[local_row] = native_recip(d*d);}
    barrier(CLK_LOCAL_MEM_FENCE);

    // get the denominator from local to register
    int index = global_row*angles+global_col;
    float d   = locald[local_row];
    float mag;
    
    float2 a = in[index];
    mag = native_sqrt(a.x*a.x+a.y*a.y); // this caused an overflow and nans!

    out[index] = (float2)(a.x*d-1,0); // make this approximation to mag because a.y/a.x = 1e-9, usually
	
}