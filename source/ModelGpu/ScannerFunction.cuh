#pragma once

#include "ModelBasic/ElementaryTypes.h"

#include "SimulationData.cuh"
#include "Math.cuh"
#include "QuantityConverter.cuh"

class ScannerFunction
{
public:
    __inline__ __device__ static void processing(Cell const* sourceCell, Token* token);

private:
    __inline__ __device__ static void spiralLookupAlgorithm(Cell const* cell, Cell const* previousCell1,
        Cell const* previousCell2, int n, HashSet<Cell const*>& visitedCells);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ScannerFunction::processing(Cell const* sourceCell, Token * token)
{
    auto& tokenMem = token->memory;
    int n = tokenMem[Enums::Scanner::INOUT_CELL_NUMBER];
    auto scanCellPre1 = sourceCell;
    auto scanCellPre2 = sourceCell;
    auto cell = token->cell;
    auto scanCell = cell;
//    spiralLookupAlgorithm(scanCell, scanCellPre1, scanCellPre2, n, tag);

    //restart?
    if (n > 0 && scanCell == scanCellPre1) {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
        scanCell = cell;
        scanCellPre1 = cell;
        tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::RESTART;
    }

    //no restart? => increase cell number
    else {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = n + 1;
        tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::SUCCESS;

        //prove whether finished or not
        auto scanCellPreTemp1 = sourceCell;
        auto scanCellPreTemp2 = sourceCell;
        auto scanCellTemp = cell;
//        spiralLookupAlgorithm(scanCellTemp, scanCellPreTemp1, scanCellPreTemp2, n + 1, tag);
        if (scanCellTemp == scanCellPreTemp1) {
            tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::FINISHED;
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
        auto len = Math::length(Math::sub(scanCell->relPos, scanCellPre1->relPos));
        tokenMem[Enums::Scanner::OUT_DISTANCE] = QuantityConverter::convertShiftLenToData(len);
    }

    //further cell
    if (n > 1) {

        //calc angle from cell n to cell n-1
        auto a1 = Math::angleOfVector(Math::sub(scanCellPre2->relPos, scanCellPre1->relPos));
        auto a2 = Math::angleOfVector(Math::sub(scanCellPre1->relPos, scanCell->relPos));
        auto angle = a1 - a2;
        tokenMem[Enums::Scanner::OUT_ANGLE] = QuantityConverter::convertAngleToData(angle);

        //calc dist from cell n to cell n-1
        auto len = Math::length(Math::sub(scanCell->relPos, scanCellPre1->relPos));
        tokenMem[Enums::Scanner::OUT_DISTANCE] = QuantityConverter::convertShiftLenToData(len);
    }

    //scan cell
    auto e = floorf(scanCell->energy);
    if (e > 255) {
        e = 255;
    }
    tokenMem[Enums::Scanner::OUT_ENERGY] = e;
    tokenMem[Enums::Scanner::OUT_CELL_MAX_CONNECTIONS] = scanCell->maxConnections;
    tokenMem[Enums::Scanner::OUT_CELL_BRANCH_NO] = scanCell->branchNumber;
/*
    auto metadata = scanCell->getMetadata();
    tokenMem[Enums::Scanner::OUT_CELL_METADATA] = metadata.color;
*/
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION] = static_cast<char>(scanCell->cellFunctionType);
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA] = scanCell->numStaticBytes;
    for (int i = 0; i < scanCell->numStaticBytes; ++i) {
        tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + 1 + i] = scanCell->staticData[i];
    }
    int mutableDataIndex = scanCell->numStaticBytes + 1;
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + mutableDataIndex] = scanCell->numMutableBytes;
    for (int i = 0; i < scanCell->numMutableBytes; ++i) {
        tokenMem[Enums::Scanner::OUT_CELL_FUNCTION_DATA + mutableDataIndex + 1 + i] = scanCell->mutableData[i];
    }

    //scan cluster
    auto mass = floorf(cell->cluster->numCellPointers);
    if (mass > 255) {
        mass = 255;
    }
    tokenMem[Enums::Scanner::OUT_MASS] = mass;
}

__inline__ __device__ void ScannerFunction::spiralLookupAlgorithm(Cell const * cell, Cell const * previousCell1, 
    Cell const * previousCell2, int n, HashSet<Cell const*>& visitedCells)
{
    //tag cell
    visitedCells.insert(cell);

    //finished?
    if (n == 0) {
        return;
    }

    //calc angle from previousCell to baseCell
    auto originAngle = Math::angleOfVector(Math::sub(previousCell1->relPos, cell->relPos));

    //iterate over all connected base cells
    auto nextCellFound = false;
    Cell* nextCell = nullptr;
    auto nextCellAngle = 0.0f;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto nextCandidateCell = cell->connections[i];
        if (!visitedCells.contains(nextCandidateCell) && !nextCandidateCell->tokenBlocked) {

            //calc angle from "nextCandCell"
            auto angle = Math::angleOfVector(Math::sub(nextCandidateCell->relPos, cell->relPos));

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
        previousCell2 = previousCell1;
        previousCell1 = cell;
        cell = nextCell;
        spiralLookupAlgorithm(cell, previousCell1, previousCell2, n - 1, visitedCells);
    }

    //no next cell found? => finish
    else {
        previousCell2 = previousCell1;
        previousCell1 = cell;
    }
}
