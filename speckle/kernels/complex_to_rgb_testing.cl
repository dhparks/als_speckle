__kernel void execute(
    __global float2* in,
    __global uchar* out, //8 bit integer
    float max) // this scales v; needs to be precomputed

{	
        // before this runs, need to divide the array by the maximum value
        // of either 1. each frame or 2. the whole cube in order to properly
        // set v.
        
        // adapted from http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
        
        int i = get_global_id(0); // frame index
        int j = get_global_id(1); // row index
        int k = get_global_id(2); // column index
        
        int frames  = get_global_size(0); // number of frames
        int rows    = get_global_size(1); // number of rows
        int columns = get_global_size(2); // number of columns
        
        int index_in = i*rows*columns+j*columns+k;
        
        float re = in[index_in].x;
        float im = in[index_in].y;
        
        // convert re and im to hls
        float s = 1.0;
        float v = native_sqrt(re*re+im*im)/max;
        float h = (atan2pi(im,re)+1)*6; // 0 to 6; hi in the python function
        int  h2 = (int)(h+0.01f); // make h into an integer
        float f = h-floor(h);
        
        // make the p, q, t components
        float p = 0;
        float q = v*(1-f);
        float t = v*f;
        
        // based on the value of h; select the proper components
        float r = 0;
        float g = 0;
        float b = 0;
        if (h2 == 0) {r = v; g = t; g = p;}
        if (h2 == 1) {r = q; g = v; g = p;}
        if (h2 == 2) {r = p; g = v; g = t;}
        if (h2 == 3) {r = p; g = q; g = v;}
        if (h2 == 4) {r = t; g = p; g = v;}
        if (h2 == 5) {r = v; g = p; g = q;}
        
        // assign rgb to out. first, calculate the index.
        // this assume the following structure (frame,row,column,channel)
        int channels = 1;
        int index_out = i*(rows*columns*channels)+j*(columns*channels)+k*channels;

        out[index_out]   = v; //convert to 8-bit integers
        //out[index_out+1] = (uchar)(255*0);
        //out[index_out+2] = (uchar)(255*0);

}