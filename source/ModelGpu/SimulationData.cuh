#pragma once

#include "Base.cuh"
#include "CudaAccessTOs.cuh"
#include "Map.cuh"

struct Cell;
struct Token
{
    float energy;
    char memory[MAX_TOKEN_MEM_SIZE];
    Cell* sourceCell;
    Cell* cell;
};


struct Particle
{
    uint64_t id;
    float energy;
    float2 pos;
    float2 vel;

    //auxiliary data
    int locked;	//0 = unlocked, 1 = locked
    bool alive;
};

struct Cluster;
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

struct Cluster
{
    uint64_t id;
    float2 pos;
    float2 vel;
    float angle;
    float angularVel;
    float angularMass;
    int numCellPointers;
    Cell** cellPointers;
    int numTokenPointers;
    Token** tokenPointers;

    //auxiliary data
    bool decompositionRequired;
    int locked;	//0 = unlocked, 1 = locked
    Cluster* clusterToFuse;
};

struct SimulationData
{
    int2 size;

    Cell **cellMap;
    Particle **particleMap;

    ArrayController<Cluster> clusters;
    ArrayController<Cluster> clustersNew;
    ArrayController<Cell*> cellPointers;
    ArrayController<Cell> cells;
    ArrayController<Particle> particles;
    ArrayController<Particle> particlesNew;
    ArrayController<Token*> tokenPointers;
    ArrayController<Token> tokens;

    CudaNumberGenerator numberGen;
};

