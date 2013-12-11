__kernel void execute(
    __global float* image,
    __global float* out,
    __local  float* localmem)

{

    // this defines the change of coordinates between the local worker index
    // and the corresponding pixel in localmem
    #define localindex(i,j,lw) (lw+2)*(j+1)+(i+1)

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
    
    localmem[localindex(li,lj,ln)] = image[gx];

    int pad    = 1;
    int gj_pad = gj;
    int gi_pad = gi;

    // bottom row
    if (lj == 0) {
      gj_pad = gj-1;
      if (gj_pad < 0) {gj_pad = gj_pad+gn;}
      localmem[li+1] = image[gi+gj_pad*gn];
    }
    
    // top row
    if (lj == ln-1) {
      gj_pad = gj+1;
      if (gj_pad >= gn) {gj_pad = gj_pad-gn;}
      localmem[li+1+(ln+1)*(ln+2)] = image[gi+gj_pad*gn];
    }
    
    // left side
    if (li == 0) {
      gi_pad = gi-1;
      if (gi_pad < 0) {gi_pad = gi_pad+gn;}
      localmem[(ln+2)*(lj+1)] = image[gi_pad+gj*gn];
    }
    
    // right side
    if (li == ln-1) {
      gi_pad = gi+1;
      if (gi_pad >= gn) {gi_pad = gi_pad-gn;}
      localmem[(ln+2)*(lj+2)-1] = image[gi_pad+gj*gn];
    }
    
    // corners
    float tmp = image[gi_pad+gj_pad*gn];
    if (lj == 0 && li == 0)       {localmem[0]               = tmp;}
    if (lj == 0 && li == ln-1)    {localmem[ln+1]            = tmp;}
    if (lj == ln-1 && li == 0)    {localmem[(ln+1)*(ln+2)]   = tmp;}
    if (lj == ln-1 && li == ln-1) {localmem[(ln+2)*(ln+2)-1] = tmp;}
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // now that the memory is copied from global into local, each
    // worker can access it quickly. pull the necessary elements
    // for the bitonic sort into registers.
    
    float r0 = localmem[localindex(li-1,lj-1,ln)];
    float r1 = localmem[localindex(li+0,lj-1,ln)];
    float r2 = localmem[localindex(li+1,lj-1,ln)];
    float r3 = localmem[localindex(li-1,lj+0,ln)];
    float r4 = localmem[localindex(li+0,lj+0,ln)];
    float r5 = localmem[localindex(li+1,lj+0,ln)];
    float r6 = localmem[localindex(li-1,lj+1,ln)];
    float r7 = localmem[localindex(li+0,lj+1,ln)];
    float r8 = localmem[localindex(li+1,lj+1,ln)];
    
    // now run the bitonic sort
    float swap_min = 0.0f;
    float swap_max = 0.0f;
    
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

    out[gx] = r4;
}