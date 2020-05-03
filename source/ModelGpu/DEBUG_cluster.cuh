#pragma once

#include "Cluster.cuh"
#include "Token.cuh"
#include "ConstantMemory.cuh"

#define STOP(a, b) printf("parameter: %d, %d\n", a, b); while(true) {};

class DEBUG_cluster
{
public:
    __inline__ __device__ static void calcEnergy_block(Cluster* cluster, float& result)
    {
        if (0 == threadIdx.x) {
            atomicAdd_block(&result, Physics::kineticEnergy(cluster->numCellPointers, cluster->vel, cluster->angularMass, cluster->angularVel));
        }
        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            atomicAdd_block(&result, cell->getEnergy());
        }
        auto const tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            atomicAdd_block(&result, token->getEnergy());
        }
    }

    __inline__ __device__ static void check_block(SimulationData* data, Cluster* cluster, int a, int b = -1)
    {
        if (0 == threadIdx.x) {
            if (!checkPointer(cluster, data->entities.clusters)) {
                printf("wrong cluster pointer\n");
                STOP(a, b);
            }
        }
        __syncthreads();
        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];

            if (!checkPointer(cluster->cellPointers + cellIndex, data->entities.cellPointers)) {
                printf("wrong cell pointer pointer\n");
                STOP(a, b);
            }

            if (!checkPointer(cell, data->entities.cells)) {
                printf("wrong cell pointer\n");
                STOP(a, b);
            }

            for (int otherCellIndex = cellIndex + 1; otherCellIndex < cluster->numCellPointers; ++otherCellIndex) {
                auto const& otherCell = cluster->cellPointers[otherCellIndex];
                if (cell == otherCell) {
                    printf("cells are not unique within cluster\n");
                    STOP(a, b)
                }
            }

            if (cell->getEnergy() < 0) {
                printf("negative cell energy: %f\n", cell->getEnergy());
                STOP(a, b)
            }
            if (cell->getEnergy() > 100000000) {
                printf("cell energy too high: %f\n", cell->getEnergy());
                STOP(a, b)
            }
            if (isnan(cell->getEnergy())) {
                printf("nan cell energy: %f\n", cell->getEnergy());
                STOP(a, b)
            }
            if (cell->numConnections > cell->maxConnections) {
                printf("numConnections > maxConnections\n");
                STOP(a, b)
            }

            if (cell->maxConnections > cudaSimulationParameters.cellMaxBonds) {
                printf("maxConnections > cellMaxBonds\n");
                STOP(a, b)
            }

            if (cell->cluster != cluster) {
                printf("cell is from different cluster\n");
                STOP(a, b)
            }

            if (cell->numStaticBytes > MAX_CELL_STATIC_BYTES) {
                printf("numStaticBytes too large\n");
            }

            if (cell->numMutableBytes > MAX_CELL_MUTABLE_BYTES) {
                printf("numMutableBytes too large\n");
            }

            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectingCell = cell->connections[i];
                if (connectingCell->cluster != cluster) {
                    printf("connecting cell is from different cluster\n");
                    STOP(a, b)
                }
                bool found = false;
                for (int j = 0; j < connectingCell->numConnections; ++j) {
                    auto const& connectingConnectingCell = connectingCell->connections[j];
                    if (connectingConnectingCell == cell) {
                        found = true;
                    }

                    if (!checkPointer(connectingConnectingCell, data->entities.cells)) {
                        printf("wrong cell pointer pointer\n");
                        STOP(a, b);
                    }

                }
                if (!found) {
                    printf("cells are only connected in one way\n");
                    STOP(a, b)
                }
            }
        }

        auto const tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            if (!checkPointer(cluster->tokenPointers + tokenIndex, data->entities.tokenPointers)) {
                printf("wrong token pointer pointer\n");
                STOP(a, b);
            }
            if (!checkPointer(token, data->entities.tokens)) {
                printf("wrong token pointer\n");
                STOP(a, b);
            }
            if (token->getEnergy() < 0) {
                printf("negative token energy: %f\n", token->getEnergy());
                STOP(a, b)
            }
            if (token->getEnergy() > 100000000) {
                printf("token energy too high: %f\n", token->getEnergy());
                STOP(a, b)
            }
            if (isnan(token->getEnergy())) {
                printf("nan token energy: %f\n", token->getEnergy());
                STOP(a, b)
            }
        }
    }

    template<typename T>
    __inline__ __device__ static bool checkPointer(T* pointer, Array<T> array)
    {
        if (array.getArrayForDevice() <= pointer && pointer < (array.getArrayForDevice() + array.getNumEntries())) {
            return true;
        }
        else {
            printf(
                "boundary check failed. pointer %llu not in interval [%llu, %llu). Array size: %d\n",
                (uintptr_t)(pointer),
                (uintptr_t)(array.getArrayForDevice()),
                (uintptr_t)(array.getArrayForDevice() + array.getNumEntries()),
                array.getNumEntries());
            return false;
        }
    }
};
