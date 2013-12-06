__kernel void execute(
    __global float* available,
    __global uchar* pospins,
    __global uchar* negpins,
    __global uchar* walls)

    {	
	int i = get_global_id(0);
	available[i] = (float)(walls[i]*pospins[i]*negpins[i]);
    }