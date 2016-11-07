#include "aliencellfunctionscanner.h"
#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"
#include "model/physics/physics.h"
#include "global/global.h"

#include <QString>
#include <QtCore/qmath.h>

AlienCellFunctionScanner::AlienCellFunctionScanner(AlienCell* cell,AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{
}

AlienCellFunctionScanner::AlienCellFunctionScanner (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

AlienCellFunctionScanner::AlienCellFunctionScanner (AlienCell* cell, QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

namespace {

    void spiralLookupAlgorithm (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag)
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
        AlienCell* nextCell = 0;
        qreal nextCellAngle = 0.0;
        int numCon = cell->getNumConnections();
        for( int i = 0; i < numCon; ++i ) {
            AlienCell* nextCandCell = cell->getConnection(i);
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

AlienCell::ProcessingResult AlienCellFunctionScanner::process (AlienToken* token, AlienCell* previousCell)
{
    AlienCell::ProcessingResult processingResult = _cell->process(token, previousCell);
    int n = token->memory[static_cast<int>(SCANNER::INOUT_CELL_NUMBER)];
    quint64 tag(GlobalFunctions::getTag());
    AlienCell* scanCellPre1 = previousCell;
    AlienCell* scanCellPre2 = previousCell;
    AlienCell* scanCell = _cell;
    spiralLookupAlgorithm(scanCell, scanCellPre1, scanCellPre2, n, tag);

    //restart?
    if( (n>0) && (scanCell == scanCellPre1) ) {
        token->memory[static_cast<int>(SCANNER::INOUT_CELL_NUMBER)] = 1;
        scanCell = _cell;
        scanCellPre1 = _cell;
        token->memory[static_cast<int>(SCANNER::OUT)] = static_cast<int>(SCANNER_OUT::RESTART);
    }

    //no restart? => increase cell number
    else {
        token->memory[static_cast<int>(SCANNER::INOUT_CELL_NUMBER)] = n+1;
        token->memory[static_cast<int>(SCANNER::OUT)] = static_cast<int>(SCANNER_OUT::SUCCESS);

        //prove whether finished or not
        tag = GlobalFunctions::getTag();
        AlienCell* scanCellPreTemp1 = previousCell;
        AlienCell* scanCellPreTemp2 = previousCell;
        AlienCell* scanCellTemp = _cell;
        spiralLookupAlgorithm(scanCellTemp, scanCellPreTemp1, scanCellPreTemp2, n+1, tag);
        if( scanCellTemp == scanCellPreTemp1 )
            token->memory[static_cast<int>(SCANNER::OUT)] = static_cast<int>(SCANNER_OUT::FINISHED);
    }

    //start cell
    if( n == 0 ) {
        token->memory[static_cast<int>(SCANNER::OUT_DISTANCE)] = 0;
    }

    //second cell
    if( n == 1 ) {
        token->memory[static_cast<int>(SCANNER::OUT_ANGLE)] = 0;

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        token->memory[static_cast<int>(SCANNER::OUT_DISTANCE)] = convertShiftLenToData(len);
    }

    //further cell
    if( n > 1 ) {

        //calc angle from cell n to cell n-1
        qreal a1 = Physics::angleOfVector(scanCellPre2->getRelPos() - scanCellPre1->getRelPos());
        qreal a2 = Physics::angleOfVector(-scanCell->getRelPos() + scanCellPre1->getRelPos());
        qreal angle = a1 - a2;
        token->memory[static_cast<int>(SCANNER::OUT_ANGLE)] = convertAngleToData(angle);
//        qDebug("-> v: %f, n: %f", -angle, convertDataToAngle(convertAngleToData(-angle)));

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        token->memory[static_cast<int>(SCANNER::OUT_DISTANCE)] = convertShiftLenToData(len);
    }

    //scan cell
    quint32 e = qFloor(scanCell->getEnergy());
    if( e > 255 )        //restrict value to 8 bit
        e = 255;
    token->memory[static_cast<int>(SCANNER::OUT_ENERGY)] = e;
    token->memory[static_cast<int>(SCANNER::OUT_CELL_MAX_CONNECTIONS)] = scanCell->getMaxConnections();
    token->memory[static_cast<int>(SCANNER::OUT_CELL_BRANCH_NO)] = scanCell->getTokenAccessNumber();
    AlienCellFunction* scanCellFunction = AlienCellDecorator::findObject<AlienCellFunction>(scanCell);
    token->memory[static_cast<int>(SCANNER::OUT_CELL_FUNCTION)] = static_cast<quint8>(scanCellFunction->getType());
    scanCellFunction->getInternalData(&token->memory[static_cast<int>(SCANNER::OUT_CELL_FUNCTION_DATA)]);

    //scan cluster
    quint32 mass = qFloor(_cell->getCluster()->getMass());
    if( mass > 255 )        //restrict value to 8 bit
        mass = 255;
    token->memory[static_cast<int>(SCANNER::OUT_MASS)] = mass;
    return processingResult;
}

