__kernel void execute(
    __global float* in_re,
    __global float* in_im,
    __global float2* m0) // the components
{	
	in_re[0] = m0[0].x;
	in_im[0] = m0[0].y;
}