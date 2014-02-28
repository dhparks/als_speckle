__kernel void execute(
    __global float* image, // input image
    __global float* out,    // filtered output
    __local  float* localmem // local memory
)
// take a sub array from the master domains image

{

    // median filter 5 -- pads by 2 to each side
    int pad = 2;

    // this defines the change in coordinates between
    // a local worker index and the location in localmem
    #define lidx(lrow,lcol,lcols) (lrow+pad)*(lcols+2*pad)+lcol+pad 
    
    // get global coordinates for current worker
    int gi  = get_global_id(0);    // global y (row)
    int gj  = get_global_id(1);    // global x (col)
    int gsi = get_global_size(0);  // global rows
    int gsj = get_global_size(1);  // global cols
    int gx  = gsj*gi+gj;
    
    // get local coordinates for current worker
    int li = get_local_id(0);     // localwork y (row)
    int lj = get_local_id(1);     // localwork x (col)
    int lsi = get_local_size(0);  // localwork rows
    int lsj = get_local_size(1);  // localwork cols
    int lx = lsj*li+lj;
    
    // this is the maximum index for localmem
    int lmimax = (lsi+2*pad)*(lsi+2*pad);
    
    // now stuff local memory. not every worker will contribute!
    // workers that do contribute write a (loop x loop) tile into localmem
    int loopsi = (lsi+2*pad)/lsi;
    if ((lsi+2*pad)%lsi > li) {loopsi++;};
    
    int loopsj = (lsj+2*pad)/lsj;
    if ((lsj+2*pad)%lsj > lj) {loopsj++;};
    
    int mi; int mj; int lmi; int gi2; int gj2;
    for (int loopi = 0; loopi < loopsi; loopi++) {
        for (int loopj = 0; loopj < loopsj; loopj++){
            mi  = li*loopi;
            mj  = lj*loopj;
            lmi = mi*(lsj+2*pad)+mj;
                
            // so we can write to this location. we just need
            // to figure out the corresponding index in image...
            
            gi2 = gi+mi-pad-li;
            gj2 = gj+mj-pad-lj;
            
            // cyclic boundary conditions
            if (gi2 < 0)   {gi2 = gi2+gsi;}
            if (gj2 < 0)   {gj2 = gj2+gsj;}
            if (gi2 > gsi) {gi2 = gi2-gsi;}
            if (gj2 > gsj) {gj2 = gj2-gsj;}

            localmem[lmi] = image[gi2*gsj+gj2];
        }
    };

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // swap registers
    float swap_min; float swap_max;
    
    // pull elements from localmem into private memory
    float r0 = localmem[lidx(li-2,lj-2,lsi)];
    float r1 = localmem[lidx(li-1,lj-2,lsi)];
    float r2 = localmem[lidx(li+0,lj-2,lsi)];
    float r3 = localmem[lidx(li+1,lj-2,lsi)];
    float r4 = localmem[lidx(li+2,lj-2,lsi)];
    
    float r5 = localmem[lidx(li-2,lj-1,lsi)];
    float r6 = localmem[lidx(li-1,lj-1,lsi)];
    float r7 = localmem[lidx(li+0,lj-1,lsi)];
    float r8 = localmem[lidx(li+1,lj-1,lsi)];
    float r9 = localmem[lidx(li+2,lj-1,lsi)];
    
    float r10 = localmem[lidx(li-2,lj+0,lsi)];
    float r11 = localmem[lidx(li-1,lj+0,lsi)];
    float r12 = localmem[lidx(li+0,lj+0,lsi)];
    float r13 = localmem[lidx(li+1,lj+0,lsi)];
    float r14 = localmem[lidx(li+2,lj+0,lsi)];
    
    float r15 = localmem[lidx(li-2,lj+1,lsi)];
    float r16 = localmem[lidx(li-1,lj+1,lsi)];
    float r17 = localmem[lidx(li+0,lj+1,lsi)];
    float r18 = localmem[lidx(li+1,lj+1,lsi)];
    float r19 = localmem[lidx(li+2,lj+1,lsi)];
    
    float r20 = localmem[lidx(li-2,lj+2,lsi)];
    float r21 = localmem[lidx(li-1,lj+2,lsi)];
    float r22 = localmem[lidx(li+0,lj+2,lsi)];
    float r23 = localmem[lidx(li+1,lj+2,lsi)];
    float r24 = localmem[lidx(li+2,lj+2,lsi)];
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

    out[gx] = r12;
    }
