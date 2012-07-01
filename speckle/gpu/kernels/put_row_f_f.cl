__kernel void execute(__global float* input, __global float* output, int offset)

{
	int i = get_global_id(0);
	output[i+offset] = input[i];
}

