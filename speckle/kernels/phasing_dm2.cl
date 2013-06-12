__kernel void execute(
    float b,
    float g,
    __global float* support,
    __global float2* psi_in,
    __global float2* psi_out,
    __global float2* dm_tmp,
    __global float2* output)

    {

	int i = get_global_id(0);
	
	// pull components into registers
	float  s  = support[i];
	float2 pi = psi_in[i];
	float2 po = psi_out[i];
	float2 dm = dm_tmp[i];
	
	// the difference map algorithm 
	float t1r = 2*po.x-b*dm.x+b*((1+g)*po.x-g*pi.x);
	float t1i = 2*po.y-b*dm.y+b*((1+g)*po.y-g*pi.y);
	
	float t2r = pi.x-b*po.x;
	float t2i = pi.y-b*po.y;
	
	float re  = s*t1r+(1-s)*t2r;
	float im  = s*t1i+(1-s)*t2i;

	
	// save
	output[i] = (float2)(re,im);
    }