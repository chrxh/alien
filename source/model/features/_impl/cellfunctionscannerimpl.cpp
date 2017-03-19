#include "cellfunctionscannerimpl.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/physics/physics.h"
#include "model/physics/codingphysicalquantities.h"
#include "global/global.h"

#include <QString>
#include <QtCore/qmath.h>

CellFunctionScannerImpl::CellFunctionScannerImpl(SimulationContext* context)
    : CellFunction(context)
{
}

namespace {

    void spiralLookupAlgorithm (Cell*& cell, Cell*& previousCell1, Cell*& previousCell2, int n, const quint64& tag)
    {
        //tag cell
        cell->setTag(tag);

        //finished?
        if( n == 0 )
            return;

        //calc angle from previousCell to baseCell
        qreal originAngle = Physics::angleOfVector(previousCell1->getRelPos() - cell->getRelPos());

        //iterate over all connected base cells
        bool nextCellFound = false;
        Cell* nextCell = 0;
        qreal nextCellAngle = 0.0;
        int numCon = cell->getNumConnections();
        for( int i = 0; i < numCon; ++i ) {
            Cell* nextCandCell = cell->getConnection(i);
            if( (nextCandCell->getTag() != tag )
             && (!nextCandCell->isTokenBlocked()) ) {
    //         && nextCandCell->isConnectedTo(previousCell1) ) {

                //calc angle from "nextCandCell"
                qreal angle = Physics::angleOfVector(nextCandCell->getRelPos() - cell->getRelPos());

                //another cell already found? => compare angles
                if( nextCellFound ) {

                    //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    //i.e. nextCellAngle > originAngle: angle\in (nextCellAngle,originAngle]
                    //nextCellAngle < originAngle: angle >= originAngle or angle < nextCellAngle
                    if( ((nextCellAngle > angle) && (angle >= originAngle))
                        || ((nextCellAngle < originAngle) && ((angle >= originAngle) || (angle < nextCellAngle))) ) {
                        nextCell = nextCandCell;
                        nextCellAngle = angle;
                    }

                }

                //no other cell found so far? => save cell and its angle
                else {
                    nextCell = nextCandCell;
                    nextCellAngle = angle;
                }
                nextCellFound = true;
            }
        }

        //next cell found?
        if( nextCellFound ) {
            previousCell2 = previousCell1;
            previousCell1 = cell;
            cell = nextCell;
            spiralLookupAlgorithm(cell, previousCell1, previousCell2, n-1, tag);
        }

        //no next cell found? => finish
        else {
            previousCell2 = previousCell1;
            previousCell1 = cell;
        }
    }

}

CellFeature::ProcessingResult CellFunctionScannerImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
	auto& tokenMem = token->getMemoryRef();
	int n = tokenMem[static_cast<int>(Enums::Scanner::INOUT_CELL_NUMBER)];
    quint64 tag(GlobalFunctions::createNewTag());
    Cell* scanCellPre1 = previousCell;
    Cell* scanCellPre2 = previousCell;
    Cell* scanCell = cell;
    spiralLookupAlgorithm(scanCell, scanCellPre1, scanCellPre2, n, tag);

    //restart?
    if( (n>0) && (scanCell == scanCellPre1) ) {
        tokenMem[static_cast<int>(Enums::Scanner::INOUT_CELL_NUMBER)] = 1;
        scanCell = cell;
        scanCellPre1 = cell;
        tokenMem[static_cast<int>(Enums::Scanner::OUT)] = static_cast<int>(Enums::ScannerOut::RESTART);
    }

    //no restart? => increase cell number
    else {
        tokenMem[static_cast<int>(Enums::Scanner::INOUT_CELL_NUMBER)] = n+1;
        tokenMem[static_cast<int>(Enums::Scanner::OUT)] = static_cast<int>(Enums::ScannerOut::SUCCESS);

        //prove whether finished or not
        tag = GlobalFunctions::createNewTag();
        Cell* scanCellPreTemp1 = previousCell;
        Cell* scanCellPreTemp2 = previousCell;
        Cell* scanCellTemp = cell;
        spiralLookupAlgorithm(scanCellTemp, scanCellPreTemp1, scanCellPreTemp2, n+1, tag);
        if( scanCellTemp == scanCellPreTemp1 )
            tokenMem[static_cast<int>(Enums::Scanner::OUT)] = static_cast<int>(Enums::ScannerOut::FINISHED);
    }

    //start cell
    if( n == 0 ) {
        tokenMem[static_cast<int>(Enums::Scanner::OUT_DISTANCE)] = 0;
    }

    //second cell
    if( n == 1 ) {
        tokenMem[static_cast<int>(Enums::Scanner::OUT_ANGLE)] = 0;

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        tokenMem[static_cast<int>(Enums::Scanner::OUT_DISTANCE)] = CodingPhysicalQuantities::convertShiftLenToData(len);
    }

    //further cell
    if( n > 1 ) {

        //calc angle from cell n to cell n-1
        qreal a1 = Physics::angleOfVector(scanCellPre2->getRelPos() - scanCellPre1->getRelPos());
        qreal a2 = Physics::angleOfVector(-scanCell->getRelPos() + scanCellPre1->getRelPos());
        qreal angle = a1 - a2;
        tokenMem[static_cast<int>(Enums::Scanner::OUT_ANGLE)] = CodingPhysicalQuantities::convertAngleToData(angle);
//        qDebug("-> v: %f, n: %f", -angle, convertDataToAngle(convertAngleToData(-angle)));

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        tokenMem[static_cast<int>(Enums::Scanner::OUT_DISTANCE)] = CodingPhysicalQuantities::convertShiftLenToData(len);
    }

    //scan cell
    quint32 e = qFloor(scanCell->getEnergy());
    if( e > 255 )        //restrict value to 8 bit
        e = 255;
    tokenMem[static_cast<int>(Enums::Scanner::OUT_ENERGY)] = e;
    tokenMem[static_cast<int>(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS)] = scanCell->getMaxConnections();
    tokenMem[static_cast<int>(Enums::Scanner::OUT_CELL_BRANCH_NO)] = scanCell->getTokenAccessNumber();
    CellFunction* scanCellFunction = scanCell->getFeatures()->findObject<CellFunction>();
    tokenMem[static_cast<int>(Enums::Scanner::OUT_CELL_FUNCTION)] = static_cast<quint8>(scanCellFunction->getType());
    QByteArray data = scanCellFunction->getInternalData();
	tokenMem.replace(static_cast<int>(Enums::Scanner::OUT_CELL_FUNCTION_DATA), data.size(), data);

    //scan cluster
    quint32 mass = qFloor(cell->getCluster()->getMass());
    if( mass > 255 )        //restrict value to 8 bit
        mass = 255;
    tokenMem[static_cast<int>(Enums::Scanner::OUT_MASS)] = mass;
    return processingResult;
}

