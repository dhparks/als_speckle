__kernel void execute(__global float2 *src, // source      (nr, nc)
                      __global float2 *dst, // destination (nf, nr, nc)
                      int n)               // frame in destination to which to copy source
    {
    
       int i_src  = get_global_id(0);
       int j_src  = get_global_id(1);
       int rows   = get_global_size(0);
       int cols   = get_global_size(1);
       int idx    = i_src*cols+j_src;
       int offset = cols*rows*n;
       
       dst[idx+offset] = src[idx];

    };