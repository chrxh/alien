#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct Cell
{
    uint64_t id;
    Cluster* cluster;
    float2 relPos;
    float2 absPos;
    float2 vel;
    float energy;
    int branchNumber;
    bool tokenBlocked;
    int maxConnections;
    int numConnections;
    Cell* connections[MAX_CELL_BONDS];
    int cellFunctionType;
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    int protectionCounter;
    bool alive;
    int tag;
};

