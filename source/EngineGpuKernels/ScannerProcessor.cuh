#pragma once
#include "cuda_runtime_api.h"

#include "EngineInterface/Enums.h"

#include "SimulationData.cuh"
#include "Math.cuh"
#include "QuantityConverter.cuh"
#include "Token.cuh"
#include "Cell.cuh"

class ScannerProcessor
{
public:
    __device__ __inline__ static void process(Token* token, SimulationData& data);

private:
    struct SpiralLookupResult
    {
        bool finish;
        Cell * cell;
        Cell * prevCell;
        Cell * prevPrevCell;
    };
    __device__ __inline__ static SpiralLookupResult
    spiralLookupAlgorithm(int depth, Cell* cell, Cell* sourceCell, SimulationData& data);

    __device__ __inline__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ScannerProcessor::process(Token* token, SimulationData& data)
{
    auto& tokenMem = token->memory;
    unsigned int n = static_cast<unsigned char>(tokenMem[Enums::Scanner_InOutCellNumber]);
    auto cell = token->cell;

    auto lookupResult = spiralLookupAlgorithm(n + 1, cell, token->sourceCell, data);

    //restart?
    if (lookupResult.finish) {
        tokenMem[Enums::Scanner_InOutCellNumber] = 0;
        lookupResult.prevPrevCell = lookupResult.prevCell;
        lookupResult.prevCell = lookupResult.cell;
        tokenMem[Enums::Scanner_Output] = Enums::ScannerOut_Finished;
    }

    //no restart? => increase cell number
    else {
        tokenMem[Enums::Scanner_InOutCellNumber] = n + 1;
        tokenMem[Enums::Scanner_Output] = Enums::ScannerOut_Success;
    }

    //start cell
    if (n == 0) {
        tokenMem[Enums::Scanner_OutDistance] = 0;
    }

    //further cell
    if (n > 0) {
        //distance from cell n-1 to cell n-2
        auto prevCellToPrevPrevCellIndex = getConnectionIndex(lookupResult.prevCell, lookupResult.prevPrevCell);
        auto distance = lookupResult.prevCell->connections[prevCellToPrevPrevCellIndex].distance;
        tokenMem[Enums::Scanner_OutDistance] = QuantityConverter::convertDistanceToData(distance);

        if (!lookupResult.finish) {
            auto prevCellToCellIndex = getConnectionIndex(lookupResult.prevCell, lookupResult.cell);

            //calc angle from cell (n-1, n-2) to (n-1, n)
            auto prevCellNumConnections = lookupResult.prevCell->numConnections;
            float angle = 0;
            int index;
            for (index = prevCellToCellIndex + 1; index < prevCellNumConnections; ++index) {
                auto const& connection = lookupResult.prevCell->connections[index];
                angle += connection.angleFromPrevious;
                if (index == prevCellToPrevPrevCellIndex) {
                    break;
                }
            }
            if (index != prevCellToPrevPrevCellIndex) {
                for (index = 0; index <= prevCellToPrevPrevCellIndex; ++index) {
                    auto const& connection = lookupResult.prevCell->connections[index];
                    angle += connection.angleFromPrevious;
                }
            }
            tokenMem[Enums::Scanner_OutAngle] = QuantityConverter::convertAngleToData(angle - 180.0f);
        }
    }

    //scan cell
    int cellEnergy = min(static_cast<int>(floorf(lookupResult.prevCell->energy)), 255);
    tokenMem[Enums::Scanner_OutEnergy] = cellEnergy;
    tokenMem[Enums::Scanner_OutCellMaxConnections] = lookupResult.prevCell->maxConnections;
    tokenMem[Enums::Scanner_OutCellBranchNumber] = lookupResult.prevCell->branchNumber;
    tokenMem[Enums::Scanner_OutCellColor] = lookupResult.prevCell->metadata.color;
    tokenMem[Enums::Scanner_OutCellFunction] = static_cast<char>(lookupResult.prevCell->getCellFunctionType());
    tokenMem[Enums::Scanner_OutCellFunctionData] = lookupResult.prevCell->numStaticBytes;
    for (int i = 0; i < lookupResult.prevCell->numStaticBytes; ++i) {
        tokenMem[Enums::Scanner_OutCellFunctionData + 1 + i] = lookupResult.prevCell->staticData[i];
    }
    int mutableDataIndex = lookupResult.prevCell->numStaticBytes + 1;
    tokenMem[Enums::Scanner_OutCellFunctionData + mutableDataIndex] = lookupResult.prevCell->numMutableBytes;
    for (int i = 0; i < lookupResult.prevCell->numMutableBytes; ++i) {
        tokenMem[Enums::Scanner_OutCellFunctionData + mutableDataIndex + 1 + i] = lookupResult.prevCell->mutableData[i];
    }
}

__device__ __inline__ auto ScannerProcessor::spiralLookupAlgorithm(int depth, Cell* cell, Cell* sourceCell, SimulationData& data) -> SpiralLookupResult
{
    SpiralLookupResult result;

    auto visitedCellData = data.processMemory.getArray<Cell*>(256 * 2);
    HashSet<Cell*, HashFunctor<Cell*>> visitedCell(depth * 2, visitedCellData);

    result.cell = cell;
    result.prevCell = sourceCell;
    result.prevPrevCell = sourceCell;
    for (int currentDepth = 0; currentDepth < depth; ++currentDepth) {
        visitedCell.insert(result.cell);

        auto posDelta = result.prevCell->absPos - result.cell->absPos;
        data.cellMap.mapDisplacementCorrection(posDelta);
        auto originAngle = Math::angleOfVector(posDelta);

        auto nextCellFound = false;
        Cell* nextCell = nullptr;
        auto nextCellAngle = 0.0f;
        for (int i = 0; i < result.cell->numConnections; ++i) {
            auto nextCandidateCell = result.cell->connections[i].cell;
            if (!visitedCell.contains(nextCandidateCell) && !nextCandidateCell->tokenBlocked) {

                //calc angle from nextCandidateCell
                auto nextPosDelta = nextCandidateCell->absPos - cell->absPos;
                data.cellMap.mapDisplacementCorrection(nextPosDelta);
                auto angle = Math::angleOfVector(nextPosDelta);

                //another cell already found? => compare angles
                if (nextCellFound) {

                    //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    //i.e. nextCellAngle > originAngle: angle\in (nextCellAngle,originAngle]
                    //nextCellAngle < originAngle: angle >= originAngle or angle < nextCellAngle
                    if ((nextCellAngle > angle && angle >= originAngle) || (nextCellAngle < originAngle && (angle >= originAngle || angle < nextCellAngle))) {
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

__device__ __inline__ int ScannerProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == otherCell) {
            return i;
        }
    }
    return 0;
}
