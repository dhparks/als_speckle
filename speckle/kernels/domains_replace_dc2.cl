__kernel void execute(
    __global float2* speckles,
    __global float2* m0) // the components
{	
	speckles[0] = m0[0];
}