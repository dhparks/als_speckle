__kernel void execute(
    __global float* image, // input image
    __global float* out,    // filtered output
    __local float* localmem // local memory
)
// take a sub array from the master domains image

{
    
    // this defines the change of coordinates between the local worker index
    // and the corresponding pixel in localmem
    #define localindex(i,j,lw) (lw+4)*(j+2)+(i+2)

    // define coordinates
    int gi = get_global_id(0);
    int gj = get_global_id(1);
    int gn = get_global_size(1);
    int gx = gn*gj+gi;
    
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int ln = get_local_size(1);
    int lx = ln*lj+li;
    
    // each worker needs to copy some data from global into local
    // workers on the edge have to copy more data than workers in the
    // center of the workgroup.

    int pad = 2; // define this as a variable to make it portable to larger sizes of median filter
    
    //localmem[(ln+2*pad)*(lj+pad)+(li+pad)] = 2.718;//(ln+2*pad)*(lj+pad)+(li+pad);//image[gx];
    localmem[(ln+2*pad)*(lj+pad)+li+pad] = image[gx];
    
    
    // use some of the workers to populate other regions of local memory.
    int gi_pad; int gj_pad; int lj_pad; int li_pad;
    
    // bottom rows
    if (li == 0 && lj < pad) {
        
        gj_pad = gj-pad;
        lj_pad = lj-pad;
        if (gj_pad < 0) {gj_pad = gj_pad+gn;}
        
        for (int k = -pad; k < ln+pad; k++) {
            gi_pad = gi+k; li_pad = li+k;
            if (gi_pad < 0)   {gi_pad = gi_pad+gn;}
            if (gi_pad >= gn) {gi_pad = gi_pad-gn;}
            localmem[(ln+2*pad)*(lj_pad+pad)+(li_pad+pad)] = image[gj_pad*gn+gi_pad];
        }
    }
    
    // top rows
    if (li == 0 && lj > ln-pad-1) {
        
        lj_pad = lj+pad;
        gj_pad = gj+pad;
        if (gj_pad >= gn) {gj_pad = gj_pad-gn;}
        
        for (int k = -pad; k < ln+pad; k++) {
            gi_pad = gi+k; li_pad = li+k;
            if (gi_pad < 0)   {gi_pad = gi_pad+gn;}
            if (gi_pad >= gn) {gi_pad = gi_pad-gn;}
            localmem[(ln+2*pad)*(lj_pad+pad)+(li_pad+pad)] = image[gj_pad*gn+gi_pad];
        }
    }
    
    // columns
    if (li < pad) {
        // left
        for (int k = -2; k < 0; k++) {
            gi_pad = gi+k; li_pad = li+k;
            if (gi_pad < 0) {gi_pad = gi_pad+gn;}
            localmem[(ln+2*pad)*(lj+pad)+(li_pad+pad)] = image[gj*gn+gi_pad];
        }
    }
        
    if (li > ln-pad-1) {
        // right
        for (int k = 1; k < pad+1; k++) {
            gi_pad = gi+k; li_pad = li+k;
            if (gi_pad >= gn) {gi_pad = gi_pad-gn;}
            localmem[(ln+2*pad)*(lj+pad)+(li_pad+pad)] = image[gj*gn+gi_pad];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // swap registers
    float swap_min; float swap_max;
    
    // pull elements from localmem into private memory
    float r0 = localmem[localindex(li-2,lj-2,ln)];
    float r1 = localmem[localindex(li-1,lj-2,ln)];
    float r2 = localmem[localindex(li+0,lj-2,ln)];
    float r3 = localmem[localindex(li+1,lj-2,ln)];
    float r4 = localmem[localindex(li+2,lj-2,ln)];
    
    float r5 = localmem[localindex(li-2,lj-1,ln)];
    float r6 = localmem[localindex(li-1,lj-1,ln)];
    float r7 = localmem[localindex(li+0,lj-1,ln)];
    float r8 = localmem[localindex(li+1,lj-1,ln)];
    float r9 = localmem[localindex(li+2,lj-1,ln)];
    
    float r10 = localmem[localindex(li-2,lj+0,ln)];
    float r11 = localmem[localindex(li-1,lj+0,ln)];
    float r12 = localmem[localindex(li+0,lj+0,ln)];
    float r13 = localmem[localindex(li+1,lj+0,ln)];
    float r14 = localmem[localindex(li+2,lj+0,ln)];
    
    float r15 = localmem[localindex(li-2,lj+1,ln)];
    float r16 = localmem[localindex(li-1,lj+1,ln)];
    float r17 = localmem[localindex(li+0,lj+1,ln)];
    float r18 = localmem[localindex(li+1,lj+1,ln)];
    float r19 = localmem[localindex(li+2,lj+1,ln)];
    
    float r20 = localmem[localindex(li-2,lj+2,ln)];
    float r21 = localmem[localindex(li-1,lj+2,ln)];
    float r22 = localmem[localindex(li+0,lj+2,ln)];
    float r23 = localmem[localindex(li+1,lj+2,ln)];
    float r24 = localmem[localindex(li+2,lj+2,ln)];
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

    //median = r12;

    out[gi+gn*gj] = r12;
}