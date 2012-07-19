__kernel void execute(
    __global float* image // image data
)
// take a sub array from the master domains image

{
    // i and j are the center coordinates
    int i = get_global_id(0);
    int j = get_global_id(1);
    int rows = get_global_size(0);
    int cols = get_global_size(1);
    
    float swap_min = 0.0f;
    float swap_max = 0.0f;
    float median = 0.0f;

    // pull the elements
    int x1 = j-1;
    int x2 = j;
    int x3 = j+1;
    
    int y1 = i-1;
    int y2 = i;
    int y3 = i+1;
    
    int nx1 = x1;
    int nx2 = x2;
    int nx3 = x3;
    
    int ny1 = y1;
    int ny2 = y2;
    int ny3 = y3;
    
    if (x1 < 0 || x1 >= cols) {nx1 = (x1+cols)%cols;}
    if (x2 < 0 || x2 >= cols) {nx2 = (x2+cols)%cols;}
    if (x3 < 0 || x3 >= cols) {nx3 = (x3+cols)%cols;}
    if (y1 < 0 || y1 >= rows) {ny1 = (y1+rows)%rows;}
    if (y2 < 0 || y2 >= rows) {ny2 = (y2+rows)%rows;}
    if (y3 < 0 || y3 >= rows) {ny3 = (y3+rows)%rows;}
    
    float r0 = image[nx1+ny1*cols];
    float r1 = image[nx2+ny1*cols];
    float r2 = image[nx3+ny1*cols];
    float r3 = image[nx1+ny2*cols];
    float r4 = image[nx2+ny2*cols];
    float r5 = image[nx3+ny2*cols];
    float r6 = image[nx1+ny3*cols];
    float r7 = image[nx2+ny3*cols];
    float r8 = image[nx3+ny3*cols];
    
    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r3);
    swap_max = fmax(r0,r3);
    r0 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r1,r2);
    swap_max = fmax(r1,r2);
    r1 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r7);
    swap_max = fmax(r4,r7);
    r4 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r5,r6);
    swap_max = fmax(r5,r6);
    r5 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r0,r7);
    swap_max = fmax(r0,r7);
    r0 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r1,r6);
    swap_max = fmax(r1,r6);
    r1 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r2,r5);
    swap_max = fmax(r2,r5);
    r2 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r3,r4);
    swap_max = fmax(r3,r4);
    r3 = swap_min;
    r4 = swap_max;

    swap_min = fmin(r0,r2);
    swap_max = fmax(r0,r2);
    r0 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r1,r3);
    swap_max = fmax(r1,r3);
    r1 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r5,r7);
    swap_max = fmax(r5,r7);
    r5 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r6);
    swap_max = fmax(r4,r6);
    r4 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r7,r8);
    swap_max = fmax(r7,r8);
    r7 = swap_min;
    r8 = swap_max;

    swap_min = fmin(r0,r4);
    swap_max = fmax(r0,r4);
    r0 = swap_min;
    r4 = swap_max;

    swap_min = fmin(r1,r5);
    swap_max = fmax(r1,r5);
    r1 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r2,r6);
    swap_max = fmax(r2,r6);
    r2 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r3,r7);
    swap_max = fmax(r3,r7);
    r3 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r0,r2);
    swap_max = fmax(r0,r2);
    r0 = swap_min;
    r2 = swap_max;

    swap_min = fmin(r1,r3);
    swap_max = fmax(r1,r3);
    r1 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r0,r1);
    swap_max = fmax(r0,r1);
    r0 = swap_min;
    r1 = swap_max;

    swap_min = fmin(r2,r3);
    swap_max = fmax(r2,r3);
    r2 = swap_min;
    r3 = swap_max;

    swap_min = fmin(r4,r6);
    swap_max = fmax(r4,r6);
    r4 = swap_min;
    r6 = swap_max;

    swap_min = fmin(r5,r7);
    swap_max = fmax(r5,r7);
    r5 = swap_min;
    r7 = swap_max;

    swap_min = fmin(r4,r5);
    swap_max = fmax(r4,r5);
    r4 = swap_min;
    r5 = swap_max;

    swap_min = fmin(r6,r7);
    swap_max = fmax(r6,r7);
    r6 = swap_min;
    r7 = swap_max;
    
    median = r4;

    image[j+rows*i] = r4;
}