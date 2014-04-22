__kernel void execute(
    __global float* r,
    float z,
    __global float2* fourier,
    __global float2* out)

    {	
	int i = get_global_id(0);
	float ri = r[i];
	float a  = native_cos(z*ri);
	float b  = native_sin(-1*z*ri);
	float c  = fourier[i].x;
	float d  = fourier[i].y;
	out[i] = (float2)(a*c-b*d,b*c+a*d);
	
    }