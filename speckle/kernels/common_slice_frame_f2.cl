__kernel void execute(__global float2 *oneFrame, __global float2 *manyFrames, int whichFrame)
        {
            
            // calculate the index of the destination pixel oneFrame(i,j)
            int rows = get_global_size(0);
            int cols = get_global_size(1);
            
            int row_dst = get_global_id(0);
            int col_dst = get_global_id(1);
            int idx_dst = row_dst*cols+col_dst;
            
            // calculate the index of the source pixel manyFrames(whichFrame,i,j)
            int offset  = whichFrame*rows*cols;
            int idx_src = idx_dst+offset;
            
            // copy
            oneFrame[idx_dst] = manyFrames[idx_src];
        }