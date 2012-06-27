__kernel void execute(
    __global float* dft_re,
    __global float* dft_im,
    __global float2* m0) 					  
    {m0[0] = (float2)(dft_re[0],dft_im[0]);}