__kernel void execute(
    __global float2* domains,
    __global float2* m0) 					  
    {m0[0] = domains[0];}