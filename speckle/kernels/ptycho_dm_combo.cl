__kernel void execute(
	__global float2* product,
	__global float2* wave,
	__global float2* out)

    {
	int i = get_global_id(0);
	out[i] = 2*product[i]-wave[i];
    }
