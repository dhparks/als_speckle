__kernel void execute(
    __global uchar* available,
    __global uchar* pospins,
    __global uchar* negpins,
    __global uchar* walls)

    {	
	int i = get_global_id(0);
	available[i] = (uchar)(walls[i]*pospins[i]*negpins[i]);
    }