__kernel void execute(
	__global float2* wave,
	__global float2* psi_out,
	__global float2* product)

    {
	int i = get_global_id(0);

	float2 w  = wave[i];
	float2 po = psi_out[i];
	float2 p  = product[i];

	wave[i] = w+po-p;
    }
