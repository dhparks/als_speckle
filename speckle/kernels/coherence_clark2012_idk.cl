__kernel void execute(
    __global float2* in1,
    __global float2* in2,
    __global float2* out)

    {
	// given in1 (fft of psi_in) and in2 (fft of psi_in_old),
	// form I^(\Delta k) in in1 and the rotated version in in2
	
	# define cm(a,b) (float2)(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x)
	# define csqr(a) float a.x*a.x+a.y*a.y

	int i = get_global_id(0);
	float2 a = in1[i];
	float2 b = in2[i];
	out[i] = (float2)(2*a.x-b.x,2*a.y-b.y);
    }