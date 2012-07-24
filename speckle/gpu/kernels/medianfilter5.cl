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
    int x0 = j-2;
    int x1 = j-1;
    int x2 = j;
    int x3 = j+1;
    int x4 = j+2;
    
    int y0 = i-2;
    int y1 = i-1;
    int y2 = i;
    int y3 = i+1;
    int y4 = i+2;
    
    int nx0 = x0;
    int nx1 = x1;
    int nx2 = x2;
    int nx3 = x3;
    int nx4 = x4;
    
    int ny0 = y0;
    int ny1 = y1;
    int ny2 = y2;
    int ny3 = y3;
    int ny4 = y4;
    
    if (x0 < 0 || x0 >= cols) {nx0 = (x0+cols)%cols;}
    if (x1 < 0 || x1 >= cols) {nx1 = (x1+cols)%cols;}
    if (x2 < 0 || x2 >= cols) {nx2 = (x2+cols)%cols;}
    if (x3 < 0 || x3 >= cols) {nx3 = (x3+cols)%cols;}
    if (x4 < 0 || x4 >= cols) {nx4 = (x4+cols)%cols;}
    if (y0 < 0 || y0 >= rows) {ny0 = (y0+cols)%rows;}
    if (y1 < 0 || y1 >= rows) {ny1 = (y1+rows)%rows;}
    if (y2 < 0 || y2 >= rows) {ny2 = (y2+rows)%rows;}
    if (y3 < 0 || y3 >= rows) {ny3 = (y3+rows)%rows;}
    if (y4 < 0 || y4 >= rows) {ny4 = (y4+cols)%rows;}
    
    float r0 = image[nx0+ny0*cols];
    float r1 = image[nx1+ny0*cols];
    float r2 = image[nx2+ny0*cols];
    float r3 = image[nx3+ny0*cols];
    float r4 = image[nx4+ny0*cols];
    float r5 = image[nx0+ny1*cols];
    float r6 = image[nx1+ny1*cols];
    float r7 = image[nx2+ny1*cols];
    float r8 = image[nx3+ny1*cols];
    float r9 = image[nx4+ny1*cols];
    float r10 = image[nx0+ny2*cols];
    float r11 = image[nx1+ny2*cols];
    float r12 = image[nx2+ny2*cols];
    float r13 = image[nx3+ny2*cols];
    float r14 = image[nx4+ny2*cols];
    float r15 = image[nx0+ny3*cols];
    float r16 = image[nx1+ny3*cols];
    float r17 = image[nx2+ny3*cols];
    float r18 = image[nx3+ny3*cols];
    float r19 = image[nx4+ny3*cols];
    float r20 = image[nx0+ny4*cols];
    float r21 = image[nx1+ny4*cols];
    float r22 = image[nx2+ny4*cols];
    float r23 = image[nx3+ny4*cols];
    float r24 = image[nx4+ny4*cols];
    
    // run the sorting network. the median is r12
    
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

    swap_min = fmin(r8,r9);
    swap_max = fmax(r8,r9);
    r8 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r10,r11);
    swap_max = fmax(r10,r11);
    r10 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r8,r11);
    swap_max = fmax(r8,r11);
    r8 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r9,r10);
    swap_max = fmax(r9,r10);
    r9 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r8,r9);
    swap_max = fmax(r8,r9);
    r8 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r10,r11);
    swap_max = fmax(r10,r11);
    r10 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r12,r13);
    swap_max = fmax(r12,r13);
    r12 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r14,r15);
    swap_max = fmax(r14,r15);
    r14 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r15);
    swap_max = fmax(r12,r15);
    r12 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r13,r14);
    swap_max = fmax(r13,r14);
    r13 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r14,r15);
    swap_max = fmax(r14,r15);
    r14 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r13);
    swap_max = fmax(r12,r13);
    r12 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r8,r15);
    swap_max = fmax(r8,r15);
    r8 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r9,r14);
    swap_max = fmax(r9,r14);
    r9 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r10,r13);
    swap_max = fmax(r10,r13);
    r10 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r11,r12);
    swap_max = fmax(r11,r12);
    r11 = swap_min;
    r12 = swap_max;

    swap_min = fmin(r13,r15);
    swap_max = fmax(r13,r15);
    r13 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r14);
    swap_max = fmax(r12,r14);
    r12 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r14,r15);
    swap_max = fmax(r14,r15);
    r14 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r13);
    swap_max = fmax(r12,r13);
    r12 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r8,r10);
    swap_max = fmax(r8,r10);
    r8 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r9,r11);
    swap_max = fmax(r9,r11);
    r9 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r10,r11);
    swap_max = fmax(r10,r11);
    r10 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r8,r9);
    swap_max = fmax(r8,r9);
    r8 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r0,r15);
    swap_max = fmax(r0,r15);
    r0 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r1,r14);
    swap_max = fmax(r1,r14);
    r1 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r2,r13);
    swap_max = fmax(r2,r13);
    r2 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r3,r12);
    swap_max = fmax(r3,r12);
    r3 = swap_min;
    r12 = swap_max;

    swap_min = fmin(r4,r11);
    swap_max = fmax(r4,r11);
    r4 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r5,r10);
    swap_max = fmax(r5,r10);
    r5 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r6,r9);
    swap_max = fmax(r6,r9);
    r6 = swap_min;
    r9 = swap_max;

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

    swap_min = fmin(r11,r15);
    swap_max = fmax(r11,r15);
    r11 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r10,r14);
    swap_max = fmax(r10,r14);
    r10 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r9,r13);
    swap_max = fmax(r9,r13);
    r9 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r8,r12);
    swap_max = fmax(r8,r12);
    r8 = swap_min;
    r12 = swap_max;

    swap_min = fmin(r9,r11);
    swap_max = fmax(r9,r11);
    r9 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r8,r10);
    swap_max = fmax(r8,r10);
    r8 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r8,r9);
    swap_max = fmax(r8,r9);
    r8 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r10,r11);
    swap_max = fmax(r10,r11);
    r10 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r13,r15);
    swap_max = fmax(r13,r15);
    r13 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r14);
    swap_max = fmax(r12,r14);
    r12 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r12,r13);
    swap_max = fmax(r12,r13);
    r12 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r14,r15);
    swap_max = fmax(r14,r15);
    r14 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r16,r17);
    swap_max = fmax(r16,r17);
    r16 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r18,r19);
    swap_max = fmax(r18,r19);
    r18 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r16,r19);
    swap_max = fmax(r16,r19);
    r16 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r17,r18);
    swap_max = fmax(r17,r18);
    r17 = swap_min;
    r18 = swap_max;

    swap_min = fmin(r16,r17);
    swap_max = fmax(r16,r17);
    r16 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r18,r19);
    swap_max = fmax(r18,r19);
    r18 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r20,r21);
    swap_max = fmax(r20,r21);
    r20 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r22,r23);
    swap_max = fmax(r22,r23);
    r22 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r23);
    swap_max = fmax(r20,r23);
    r20 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r21,r22);
    swap_max = fmax(r21,r22);
    r21 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r22,r23);
    swap_max = fmax(r22,r23);
    r22 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r21);
    swap_max = fmax(r20,r21);
    r20 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r16,r23);
    swap_max = fmax(r16,r23);
    r16 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r17,r22);
    swap_max = fmax(r17,r22);
    r17 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r18,r21);
    swap_max = fmax(r18,r21);
    r18 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r19,r20);
    swap_max = fmax(r19,r20);
    r19 = swap_min;
    r20 = swap_max;

    swap_min = fmin(r16,r18);
    swap_max = fmax(r16,r18);
    r16 = swap_min;
    r18 = swap_max;

    swap_min = fmin(r17,r19);
    swap_max = fmax(r17,r19);
    r17 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r16,r17);
    swap_max = fmax(r16,r17);
    r16 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r18,r19);
    swap_max = fmax(r18,r19);
    r18 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r21,r23);
    swap_max = fmax(r21,r23);
    r21 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r22);
    swap_max = fmax(r20,r22);
    r20 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r20,r21);
    swap_max = fmax(r20,r21);
    r20 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r22,r23);
    swap_max = fmax(r22,r23);
    r22 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r23,r24);
    swap_max = fmax(r23,r24);
    r23 = swap_min;
    r24 = swap_max;

    swap_min = fmin(r16,r20);
    swap_max = fmax(r16,r20);
    r16 = swap_min;
    r20 = swap_max;

    swap_min = fmin(r17,r21);
    swap_max = fmax(r17,r21);
    r17 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r18,r22);
    swap_max = fmax(r18,r22);
    r18 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r19,r23);
    swap_max = fmax(r19,r23);
    r19 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r22);
    swap_max = fmax(r20,r22);
    r20 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r21,r23);
    swap_max = fmax(r21,r23);
    r21 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r22,r23);
    swap_max = fmax(r22,r23);
    r22 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r21);
    swap_max = fmax(r20,r21);
    r20 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r16,r18);
    swap_max = fmax(r16,r18);
    r16 = swap_min;
    r18 = swap_max;

    swap_min = fmin(r17,r19);
    swap_max = fmax(r17,r19);
    r17 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r18,r19);
    swap_max = fmax(r18,r19);
    r18 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r16,r17);
    swap_max = fmax(r16,r17);
    r16 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r7,r24);
    swap_max = fmax(r7,r24);
    r7 = swap_min;
    r24 = swap_max;

    swap_min = fmin(r8,r23);
    swap_max = fmax(r8,r23);
    r8 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r9,r22);
    swap_max = fmax(r9,r22);
    r9 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r10,r21);
    swap_max = fmax(r10,r21);
    r10 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r11,r20);
    swap_max = fmax(r11,r20);
    r11 = swap_min;
    r20 = swap_max;

    swap_min = fmin(r12,r19);
    swap_max = fmax(r12,r19);
    r12 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r13,r18);
    swap_max = fmax(r13,r18);
    r13 = swap_min;
    r18 = swap_max;

    swap_min = fmin(r14,r17);
    swap_max = fmax(r14,r17);
    r14 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r15,r16);
    swap_max = fmax(r15,r16);
    r15 = swap_min;
    r16 = swap_max;

    swap_min = fmin(r0,r8);
    swap_max = fmax(r0,r8);
    r0 = swap_min;
    r8 = swap_max;

    swap_min = fmin(r1,r9);
    swap_max = fmax(r1,r9);
    r1 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r2,r10);
    swap_max = fmax(r2,r10);
    r2 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r3,r11);
    swap_max = fmax(r3,r11);
    r3 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r4,r12);
    swap_max = fmax(r4,r12);
    r4 = swap_min;
    r12 = swap_max;

    swap_min = fmin(r5,r13);
    swap_max = fmax(r5,r13);
    r5 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r6,r14);
    swap_max = fmax(r6,r14);
    r6 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r7,r15);
    swap_max = fmax(r7,r15);
    r7 = swap_min;
    r15 = swap_max;

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

    swap_min = fmin(r8,r12);
    swap_max = fmax(r8,r12);
    r8 = swap_min;
    r12 = swap_max;

    swap_min = fmin(r9,r13);
    swap_max = fmax(r9,r13);
    r9 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r10,r14);
    swap_max = fmax(r10,r14);
    r10 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r11,r15);
    swap_max = fmax(r11,r15);
    r11 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r8,r10);
    swap_max = fmax(r8,r10);
    r8 = swap_min;
    r10 = swap_max;

    swap_min = fmin(r9,r11);
    swap_max = fmax(r9,r11);
    r9 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r8,r9);
    swap_max = fmax(r8,r9);
    r8 = swap_min;
    r9 = swap_max;

    swap_min = fmin(r10,r11);
    swap_max = fmax(r10,r11);
    r10 = swap_min;
    r11 = swap_max;

    swap_min = fmin(r12,r14);
    swap_max = fmax(r12,r14);
    r12 = swap_min;
    r14 = swap_max;

    swap_min = fmin(r13,r15);
    swap_max = fmax(r13,r15);
    r13 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r12,r13);
    swap_max = fmax(r12,r13);
    r12 = swap_min;
    r13 = swap_max;

    swap_min = fmin(r14,r15);
    swap_max = fmax(r14,r15);
    r14 = swap_min;
    r15 = swap_max;

    swap_min = fmin(r16,r24);
    swap_max = fmax(r16,r24);
    r16 = swap_min;
    r24 = swap_max;

    swap_min = fmin(r19,r23);
    swap_max = fmax(r19,r23);
    r19 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r18,r22);
    swap_max = fmax(r18,r22);
    r18 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r17,r21);
    swap_max = fmax(r17,r21);
    r17 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r16,r20);
    swap_max = fmax(r16,r20);
    r16 = swap_min;
    r20 = swap_max;

    swap_min = fmin(r17,r19);
    swap_max = fmax(r17,r19);
    r17 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r16,r18);
    swap_max = fmax(r16,r18);
    r16 = swap_min;
    r18 = swap_max;

    swap_min = fmin(r16,r17);
    swap_max = fmax(r16,r17);
    r16 = swap_min;
    r17 = swap_max;

    swap_min = fmin(r18,r19);
    swap_max = fmax(r18,r19);
    r18 = swap_min;
    r19 = swap_max;

    swap_min = fmin(r21,r23);
    swap_max = fmax(r21,r23);
    r21 = swap_min;
    r23 = swap_max;

    swap_min = fmin(r20,r22);
    swap_max = fmax(r20,r22);
    r20 = swap_min;
    r22 = swap_max;

    swap_min = fmin(r20,r21);
    swap_max = fmax(r20,r21);
    r20 = swap_min;
    r21 = swap_max;

    swap_min = fmin(r22,r23);
    swap_max = fmax(r22,r23);
    r22 = swap_min;
    r23 = swap_max;

    median = r12;

    image[j+rows*i] = median;
}