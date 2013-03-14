__kernel void execute(
    __global float* image,    // image data
    __global float* filtered, // image data which has been median filtered
    float threshold)         // data/medfiltered_data > threshold is considered "hot")   

// take a sub array from the master domains image

{
    // i and j are the center coordinates
    int i = get_global_id(0);
    
    int rows = get_global_size(0);
    
    float ii = image[i];
    float fi = filtered[i];

    if (ii/fi >= threshold) {image[i] = fi;}
}