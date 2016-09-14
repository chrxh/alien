#include "aliencellfunctionscanner.h"
#include "aliencellfunctionfactory.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"
#include "../physics/physics.h"
#include "../../globaldata/globalfunctions.h"

#include <QString>
#include <QtCore/qmath.h>

AlienCellFunctionScanner::AlienCellFunctionScanner()
{
}

AlienCellFunctionScanner::AlienCellFunctionScanner (quint8* cellTypeData)
{

}

AlienCellFunctionScanner::AlienCellFunctionScanner (QDataStream& stream)
{

}

void AlienCellFunctionScanner::execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose)
{
    int n = token->memory[static_cast<int>(SCANNER::INOUT_CELL_NUMBER)];
    quint64 tag(GlobalFunctions::getTag());
    AlienCell* scanCellPre1 = previousCell;
    AlienCell* scanCellPre2 = previousCell;
    AlienCell* scanCell = cell;
    spiralLookupAlgorithm(scanCell, scanCellPre1, scanCellPre2, n, tag);

    //restart?
    if( (n>0) && (scanCell == scanCellPre1) ) {
        token->memory[static_cast<int>(SCANNER::INOUT_CELL_NUMBER)] = 1;
        scanCell = cell;
        scanCellPre1 = cell;
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
        AlienCell* scanCellTemp = cell;
        spiralLookupAlgorithm(scanCellTemp, scanCellPreTemp1, scanCellPreTemp2, n+1, tag);
        if( scanCellTemp == scanCellPreTemp1 )
            token->memory[static_cast<int>(SCANNER::OUT)] = static_cast<int>(SCANNER_OUT::FINISHED);
    }

    //start cell
    if( n == 0 ) {
        token->memory[static_cast<int>(SCANNER::OUT_DIST)] = 0;
    }

    //second cell
    if( n == 1 ) {
        token->memory[static_cast<int>(SCANNER::OUT_ANGLE)] = 0;

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        token->memory[static_cast<int>(SCANNER::OUT_DIST)] = convertShiftLenToData(len);
    }

    //further cell
    if( n > 1 ) {

        //calc angle from cell n to cell n-1
        qreal a1 = Physics::calcAngle(scanCellPre2->getRelPos() - scanCellPre1->getRelPos());
        qreal a2 = Physics::calcAngle(-scanCell->getRelPos() + scanCellPre1->getRelPos());
        qreal angle = a2 - a1;
        token->memory[static_cast<int>(SCANNER::OUT_ANGLE)] = convertAngleToData(-angle);
//        qDebug("-> v: %f, n: %f", -angle, convertDataToAngle(convertAngleToData(-angle)));

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPos() - scanCellPre1->getRelPos()).length();
        token->memory[static_cast<int>(SCANNER::OUT_DIST)] = convertShiftLenToData(len);
    }

    //scan cell
    quint32 e = qFloor(scanCell->getEnergy());
    if( e > 255 )        //restrict value to 8 bit
        e = 255;
    token->memory[static_cast<int>(SCANNER::OUT_ENERGY)] = e;
    token->memory[static_cast<int>(SCANNER::OUT_CELL_MAX_CONNECTIONS)] = scanCell->getMaxConnections();
    token->memory[static_cast<int>(SCANNER::OUT_CELL_BRANCH_NO)] = scanCell->getTokenAccessNumber();
    token->memory[static_cast<int>(SCANNER::OUT_CELL_FUNCTION)] = convertCellTypeNameToNumber(scanCell->getCellFunction()->getCellFunctionName());
    scanCell->getCellFunction()->getInternalData(&token->memory[static_cast<int>(SCANNER::OUT_CELL_FUNCTION_DATA)]);

    //scan cluster
    quint32 mass = qFloor(cell->getCluster()->getMass());
    if( mass > 255 )        //restrict value to 8 bit
        mass = 255;
    token->memory[static_cast<int>(SCANNER::OUT_MASS)] = mass;
}

QString AlienCellFunctionScanner::getCellFunctionName ()
{
    return "SCANNER";
}

void AlienCellFunctionScanner::serialize (QDataStream& stream)
{
    AlienCellFunction::serialize(stream);
}

void AlienCellFunctionScanner::spiralLookupAlgorithm (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag)
{
    //tag cell
    cell->setTag(tag);

    //finished?
    if( n == 0 )
        return;

    //calc angle from previousCell to baseCell
    qreal originAngle = Physics::calcAngle(previousCell1->getRelPos() - cell->getRelPos());

    //iterate over all connected base cells
    bool nextCellFound = false;
    AlienCell* nextCell = 0;
    qreal nextCellAngle = 0.0;
    int numCon = cell->getNumConnections();
    for( int i = 0; i < numCon; ++i ) {
        AlienCell* nextCandCell = cell->getConnection(i);
        if( (nextCandCell->getTag() != tag )
         && (!nextCandCell->blockToken()) ) {
//         && nextCandCell->isConnectedTo(previousCell1) ) {

            //calc angle from "nextCandCell"
            qreal angle = Physics::calcAngle(nextCandCell->getRelPos() - cell->getRelPos());

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

int AlienCellFunctionScanner::convertCellTypeNameToNumber (QString type)
{
    if( type == "COMPUTER" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::COMPUTER);
    if( type == "PROPULSION" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::PROP);
    if( type == "SCANNER" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::SCANNER);
    if( type == "WEAPON" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::WEAPON);
    if( type == "CONSTRUCTOR" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::CONSTR);
    if( type == "SENSOR" )
        return static_cast<int>(SCANNER_OUT_CELL_TYPE::SENSOR);
    return 0;
}

/*
//modified breadth-first search algorithm
void AlienCellFunctionScanner::scan2 (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag)
{
    //tag cell
    cell->setTag(tag);

    //finished?
    if( n == 0 )
        return;


    //1.step: try find all connected surrounding cells
    //-------
    AlienCell* baseCell = cell;
    int numCon = baseCell->getNumConnections();
    bool nextCellFound = false;
    bool firstTime = true;
    do {

        //calc angle from previousCell to baseCell
        AlienCell* refCell = previousCell1;
        if( firstTime )
            firstTime = false;
        else
            refCell = cell;
        qreal originAngle = Physics::calcAngle(refCell->getRelPos() - baseCell->getRelPos());
        nextCellFound = false;
        AlienCell* nextCell = 0;
        qreal nextCellAngle = 0.0;

        //iterate over all connected base cells which are connected to previousCell1
        for( int i = 0; i < numCon; ++i ) {
            AlienCell* nextCandCell = baseCell->getConnection(i);
            if( (nextCandCell->getTag() != tag )
             && (!nextCandCell->blockToken())
             && (nextCandCell->isConnectedTo(refCell) || (nextCandCell == refCell)) ) {

                //calc angle from "nextCandCell"
                qreal angle = Physics::calcAngle(nextCandCell->getRelPos() - baseCell->getRelPos());

                //another cell already found? => compare angles
                if( nextCellFound ) {

                    //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    //i.e. nextCellAngle >= originAngle: angle\in (nextCellAngle,angle)
                    //nextCellAngle < originAngle: angle >= originAngle or angle <= nextCellAngle
                    if( ((nextCellAngle > angle) && (angle >= originAngle))
                        || ((nextCellAngle < originAngle) && ((angle >= originAngle) || (angle <= nextCellAngle))) ) {
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
            nextCell->setTag(tag);
            n = n - 1;
            if( n == 0 )
                return;
        }
    }
    while(nextCellFound);

    //2.step: jump to next cell
    //-------
    nextCellFound = false;
    AlienCell* nextCell = 0;
    qreal nextCellAngle = 0.0;

    //calc angle from previous cell
    qreal originAngle = Physics::calcAngle(previousCell1->getRelPos() - cell->getRelPos());

    //iterate over all connected base cells which are connected to previousCell1
    numCon = cell->getNumConnections();
    for( int i = 0; i < numCon; ++i ) {
        AlienCell* nextCandCell = cell->getConnection(i);
        if( (nextCandCell->getTag() != tag )
         && (!nextCandCell->blockToken()) ) {
         //&& (nextCandCell->isConnectedTo(previousCell1) || (nextCandCell == previousCell1)) ) {

            //calc angle from "nextCandCell"
            qreal angle = Physics::calcAngle(-nextCandCell->getRelPos() + cell->getRelPos());  //note the reverse sign

            //another cell already found? => compare angles
            if( nextCellFound ) {

                //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                //i.e. nextCellAngle >= originAngle: angle\in (nextCellAngle,angle)
                //nextCellAngle < originAngle: angle >= originAngle or angle <= nextCellAngle
                if( ((nextCellAngle > angle) && (angle >= originAngle))
                    || ((nextCellAngle < originAngle) && ((angle >= originAngle) || (angle <= nextCellAngle))) ) {
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
        scan2(cell, previousCell1, previousCell2, n-1, tag);
        return;
    }

    //3.step: no next cell found? => indicate finish
    //-------
    previousCell2 = previousCell1;
    previousCell1 = cell;
}

//breadth-first search algorithm
void AlienCellFunctionScanner::scan (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag)
{
    //finish?
    if( n == 0 )
        return;

    //not finish? => tag cell
    else
        cell->setTag(tag);

    //calc angle from origin cell
    qreal originAngle = Physics::calcAngle(previousCell1->getRelPos() - cell->getRelPos());

    //determine cell which is connected to "cell" and "previousCell1" and has minimal angle which is "larger" than originAngle
    bool found = false;
    AlienCell* nextCell = 0;
    qreal nextCellAngle = 0.0;
    int numCon = cell->getNumConnections();
    for( int i = 0; i < numCon; ++i ) {
        AlienCell* newCell = cell->getConnection(i);
        if( (newCell->getTag() != tag ) && (!newCell->blockToken()) ) {
            if( newCell->isConnectedTo(previousCell1) ) {

                //calc angle from "newCell"
                qreal angle = Physics::calcAngle(newCell->getRelPos() - cell->getRelPos());

                //another cell already found? => compare angles
                if( found ) {

                    //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    //i.e. nextCellAngle >= originAngle: angle\in (nextCellAngle,angle)
                    //nextCellAngle < originAngle: angle >= originAngle or angle <= nextCellAngle
                    if( ((nextCellAngle >= angle) && (angle >= originAngle))
                        || ((nextCellAngle < originAngle) && ((angle >= originAngle) || (angle <= nextCellAngle))) ) {
                        nextCell = newCell;
                        nextCellAngle = angle;
                    }

                }

                //no other cell found so far? => save cell and its angle
                else {
                    nextCell = newCell;
                    nextCellAngle = angle;
                }
                found = true;
            }
        }
    }

    //cell found?
    if( found ) {
        previousCell2 = previousCell1;
        previousCell1 = cell;
        cell = nextCell;
        return scan(cell, previousCell1, previousCell2, n-1, tag);
    }

    //nothing found => second round (drop "connected to previousCell1" condition)!
    found = false;
    for( int i = 0; i < numCon; ++i ) {
        AlienCell* newCell = cell->getConnection(i);
        if( (newCell->getTag() != tag ) && (!newCell->blockToken()) ) {

            //calc angle from "newCell"
            qreal angle = Physics::calcAngle(newCell->getRelPos() - cell->getRelPos());

            //another cell already found? => compare angles
            if( found ) {

                //new angle should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                //i.e. nextCellAngle >= originAngle: angle\in (nextCellAngle,angle)
                //nextCellAngle < originAngle: angle >= originAngle or angle <= nextCellAngle
                if( ((nextCellAngle >= angle) && (angle >= originAngle))
                    || ((nextCellAngle < originAngle) && ((angle >= originAngle) || (angle <= nextCellAngle))) ) {
                    nextCell = newCell;
                    nextCellAngle = angle;
                }

            }

            //no other cell found so far? => save cell and its angle
            else {
                nextCell = newCell;
                nextCellAngle = angle;
            }
            found = true;
        }
    }

    //cell found?
    if( found ) {
        previousCell2 = previousCell1;
        previousCell1 = cell;
        cell = nextCell;
        return scan(cell, previousCell1, previousCell2, n-1, tag);
    }

    //end?
    previousCell2 = previousCell1;
    previousCell1 = cell;
    return;
}

*/
