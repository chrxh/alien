#pragma once
#include "math_functions.h"

#include "ModelBasic/ElementaryTypes.h"

#include "SimulationData.cuh"
#include "Math.cuh"
#include "QuantityConverter.cuh"

class ScannerFunction
{
public:
    __inline__ __device__ static void processing(Token* token);

private:
    struct SpiralLookupResult
    {
        bool finish;
        Cell const* cell;
        Cell const* prevCell;
        Cell const* prevPrevCell;
    };
    __device__ static SpiralLookupResult spiralLookupAlgorithm(int depth, Cell const* cell, Cell const* sourceCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ScannerFunction::processing(Token * token)
{
    auto& tokenMem = token->memory;
    unsigned int n = static_cast<unsigned char>(tokenMem[Enums::Scanner::INOUT_CELL_NUMBER]);
    auto cell = token->cell;

    auto lookupResult = spiralLookupAlgorithm(n, cell, token->sourceCell);

    //restart?
    if (n > 0 && lookupResult.finish) {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
        lookupResult.cell = cell;
        lookupResult.prevCell = cell;
        tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::RESTART;
    }

    //no restart? => increase cell number
    else {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = n + 1;

        //prove whether finished or not
        auto lookupResult = spiralLookupAlgorithm(n + 1, cell, token->sourceCell);
        if (lookupResult.finish) {
            tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::FINISHED;
        }
        else {
            tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::SUCCESS;
        }
    }

    //start cell
    if (n == 0) {
        tokenMem[Enums::Scanner::OUT_DISTANCE] = 0;
    }

    //second cell
    if (n == 1) {
        tokenMem[Enums::Scanner::OUT_ANGLE] = 0;

        //calc dist from cell n to cell n-1
        auto len = Math::length(lookupResult.cell->relPos - lookupResult.prevCell->relPos);
        tokenMem[Enums::Scanner::OUT_DISTANCE] = QuantityConverter::convertDistanceToData(len);
    }

    //further cell
    if (n > 1) {

        //calc angle from cell n to cell n-1
        auto a1 = Math::angleOfVector(lookupResult.prevPrevCell->relPos - lookupResult.prevCell->relPos);
        auto a2 = Math::angleOfVector(lookupResult.prevCell->relPos - lookupResult.cell->relPos);
        auto angle = a1 - a2;
        tokenMem[Enums::Scanner::OUT_ANGLE] = QuantityConverter::convertAngleToData(angle);

        //calc dist from cell n to cell n-1
        auto len = Math::length(lookupResult.cell->relPos - lookupResult.prevCell->relPos);
        tokenMem[Enums::Scanner::OUT_DISTANCE] = QuantityConverter::convertDistanceToData(len);
    }

    //scan cell
    int cellEnergy = min(static_cast<int>(floorf(lookupResult.cell->energy)), 255);
    tokenMem[Enums::Scanner::OUT_ENERGY] = cellEnergy;
    tokenMem[Enums::Scanner::OUT_CELL_MAX_CONNECTIONS] = lookupResult.cell->maxConnections;
    tokenMem[Enums::Scanner::OUT_CELL_BRANCH_NO] = lookupResult.cell->branchNumber;
/*
    auto metadata = scanCell->getMetadata();
    tokenMem[Enums::Scanner::OUT_CELL_METADATA] = metadata.color;
*/
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION] = static_cast<char>(lookupResult.cell->cellFunctionType);
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA] = lookupResult.cell->numStaticBytes;
    for (int i = 0; i < lookupResult.cell->numStaticBytes; ++i) {
        tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + 1 + i] = lookupResult.cell->staticData[i];
    }
    int mutableDataIndex = lookupResult.cell->numStaticBytes + 1;
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + mutableDataIndex] = lookupResult.cell->numMutableBytes;
    for (int i = 0; i < lookupResult.cell->numMutableBytes; ++i) {
        tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + mutableDataIndex + 1 + i] = lookupResult.cell->mutableData[i];
    }

    //scan cluster
    auto mass = floorf(cell->cluster->numCellPointers);
    if (mass > 255) {
        mass = 255;
    }
    tokenMem[Enums::Scanner::OUT_MASS] = mass;

}

__device__ auto ScannerFunction::spiralLookupAlgorithm(int depth, Cell const* cell, Cell const* sourceCell)
    -> SpiralLookupResult
{
    SpiralLookupResult result;

    Cell const* visitedCellData[256*2];
    HashSet<Cell const*> visitedCell(depth*2, visitedCellData);

    result.cell = cell;
    result.prevCell = sourceCell;
    result.prevPrevCell = sourceCell;
    for (int currentDepth = 0; currentDepth < depth; ++currentDepth) {
        visitedCell.insert(result.cell);

        auto originAngle = Math::angleOfVector(result.prevCell->relPos - result.cell->relPos);

        auto nextCellFound = false;
        Cell* nextCell = nullptr;
        auto nextCellAngle = 0.0f;
        for (int i = 0; i < result.cell->numConnections; ++i) {
            auto nextCandidateCell = result.cell->connections[i];
            if (!visitedCell.contains(nextCandidateCell) && !nextCandidateCell->tokenBlocked) {

                //calc angle from nextCandidateCell
                auto angle = Math::angleOfVector(nextCandidateCell->relPos - cell->relPos);

                //another cell already found? => compare angles
                if (nextCellFound) {

                    //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    //i.e. nextCellAngle > originAngle: angle\in (nextCellAngle,originAngle]
                    //nextCellAngle < originAngle: angle >= originAngle or angle < nextCellAngle
                    if ((nextCellAngle > angle && angle >= originAngle)
                        || (nextCellAngle < originAngle && (angle >= originAngle || angle < nextCellAngle))) {
                        nextCell = nextCandidateCell;
                        nextCellAngle = angle;
                    }

                }

                //no other cell found so far? => save cell and its angle
                else {
                    nextCell = nextCandidateCell;
                    nextCellAngle = angle;
                }
                nextCellFound = true;
            }
        }

        //next cell found?
        if (nextCellFound) {
            result.prevPrevCell = result.prevCell;
            result.prevCell = result.cell;
            result.cell = nextCell;
        }

        //no next cell found? => finish
        else {
            result.finish = true;
            return result;
        }
    }
    result.finish = false;
    return result;
}
