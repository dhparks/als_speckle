__kernel void execute(
	__global float2* probe,
	__global float2* object,
	__global float2* product,
	__global float2* wave,
	__global float2* psi_in,
	int r0, int c0, int C)

    {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int J = get_global_size(1);
	
	int idx_psi = i*J+j;
	int idx_obj = (i+r0)*C+j+c0;

	float2 p = probe[idx_psi];
	float2 o = object[idx_obj];

	float2 z = (float2)(p.x*o.x-p.y*o.y,p.x*o.y+p.y*o.x);

	product[idx_psi] = z;
	psi_in[idx_psi] = 2*z-wave[idx_psi];
    }
