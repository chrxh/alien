#pragma once

struct CudaConstants
{
    int NUM_THREADS_PER_BLOCK = 0;/* = 64*/
    int NUM_BLOCKS = 0;/* 64*/

    int MAX_CLUSTERS = 0; /*500000*/
    int MAX_CELLS = 0;/* 2000000*/
    int MAX_PARTICLES = 0;/* 2000000*/
    int MAX_TOKENS = 0;/* 500000*/
    int MAX_CELLPOINTERS = 0;/* MAX_CELLS * 10*/
    int MAX_CLUSTERPOINTERS = 0;/* MAX_CLUSTERS * 10*/
    int MAX_PARTICLEPOINTERS = 0;/* MAX_PARTICLES * 10*/
    int MAX_TOKENPOINTERS = 0;/* MAX_TOKENS * 10*/
    int DYNAMIC_MEMORY_SIZE = 0;

    int MAX_STRINGBYTES = 0;
};
