__kernel void execute(
    __global float* support,
    __global float2* psi_out,
    __global float2* output)

    {	
	int i = get_global_id(0);
	
	// pull components into registers
	float s   = support[i];
	float2 po = psi_out[i];

	float re = s*po.x;
	float im = s*po.y;
	output[i] = (float2)(re,im);
	
    }