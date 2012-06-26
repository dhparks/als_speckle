__kernel void execute(__global float2 *dst, __global float2 *src, int x0, int y0, int rows, int cols, int n, int N)
    {
        int i_dst = get_global_id(0);
        int j_dst = get_global_id(1);
    
        // i_dst and j_dst are the coordinates of the destination. we "simply" need to turn them into 
        // the correct indices to move values from src to dst.
        
        int dst_index = (n*rows*cols)+(j_dst*rows)+i_dst; // (frames)+(rows)+cols
        int src_index = (i_dst+x0)+(j_dst+y0)*N; // (cols)+(rows)
        
        dst[dst_index] = src[src_index];
    }