__kernel void execute(__global float2 *dst, __global float2 *src, int n, int L, int N)
        {
            int i_dst = get_global_id(0);
            int j_dst = get_global_id(1);
            int x0 = N/2-L/2;
            int y0 = N/2-L/2;
        
            // i_dst and j_dst are the coordinates of the destination. we "simply" need to turn them into 
            // the correct indices to move values from src to dst.
            
            int dst_index = (n*L*L)+(j_dst*L)+i_dst; // (frames)+(rows)+cols
            int src_index = (i_dst+x0)+(j_dst+y0)*N; // (cols)+(rows)
            
            dst[dst_index] = src[src_index];
        }