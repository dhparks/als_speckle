__kernel void execute(__global float2 *dst, __global float2 *src, int n, int N, int sr0, int sr2)
        {
            // calculate the index of the dst pixel dst(i,j)
            int rows = get_global_size(0);
            int cols = get_global_size(1);
            
            int row_dst = get_global_id(0);
            int col_dst = get_global_id(1);
            int doffset = n*rows*cols;
            int idx_dst = doffset+row_dst*get_global_size(1)+col_dst; // index of pixel in dst
            
            // calculate the index of the same pixel in src        
            int row_src = row_dst+sr0;
            int col_src = col_dst+sr2; 
            int idx_src = row_src*N+col_src;

            dst[idx_dst] = src[idx_src]; // copy
        }