#include <QString>
#include <QtCore/qmath.h>

#include "global/ServiceLocator.h"
#include "global/NumberGenerator.h"
#include "model/physics/Physics.h"
#include "model/physics/CodingPhysicalQuantities.h"
#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"
#include "model/entities/Token.h"
#include "model/context/UnitContext.h"
#include "model/context/SimulationParameters.h"

#include "CellFunctionScannerImpl.h"

CellFunctionScannerImpl::CellFunctionScannerImpl(UnitContext* context)
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
        qreal originAngle = Physics::angleOfVector(previousCell1->getRelPosition() - cell->getRelPosition());

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
                qreal angle = Physics::angleOfVector(nextCandCell->getRelPosition() - cell->getRelPosition());

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
	auto numberGen = _context->getNumberGenerator();
	ProcessingResult processingResult{ false, 0 };
	auto& tokenMem = token->getMemoryRef();
	int n = tokenMem[Enums::Scanner::INOUT_CELL_NUMBER];
    quint64 tag(numberGen->getTag());
    Cell* scanCellPre1 = previousCell;
    Cell* scanCellPre2 = previousCell;
    Cell* scanCell = cell;
    spiralLookupAlgorithm(scanCell, scanCellPre1, scanCellPre2, n, tag);

    //restart?
    if( (n>0) && (scanCell == scanCellPre1) ) {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
        scanCell = cell;
        scanCellPre1 = cell;
        tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::RESTART;
    }

    //no restart? => increase cell number
    else {
        tokenMem[Enums::Scanner::INOUT_CELL_NUMBER] = n+1;
        tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::SUCCESS;

        //prove whether finished or not
        tag = numberGen->getTag();
        Cell* scanCellPreTemp1 = previousCell;
        Cell* scanCellPreTemp2 = previousCell;
        Cell* scanCellTemp = cell;
        spiralLookupAlgorithm(scanCellTemp, scanCellPreTemp1, scanCellPreTemp2, n+1, tag);
        if( scanCellTemp == scanCellPreTemp1 )
            tokenMem[Enums::Scanner::OUT] = Enums::ScannerOut::FINISHED;
    }

    //start cell
    if( n == 0 ) {
        tokenMem[Enums::Scanner::OUT_DISTANCE] = 0;
    }

    //second cell
    if( n == 1 ) {
        tokenMem[Enums::Scanner::OUT_ANGLE] = 0;

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPosition() - scanCellPre1->getRelPosition()).length();
        tokenMem[Enums::Scanner::OUT_DISTANCE] = CodingPhysicalQuantities::convertShiftLenToData(len);
    }

    //further cell
    if( n > 1 ) {

        //calc angle from cell n to cell n-1
        qreal a1 = Physics::angleOfVector(scanCellPre2->getRelPosition() - scanCellPre1->getRelPosition());
        qreal a2 = Physics::angleOfVector(-scanCell->getRelPosition() + scanCellPre1->getRelPosition());
        qreal angle = a1 - a2;
        tokenMem[Enums::Scanner::OUT_ANGLE] = CodingPhysicalQuantities::convertAngleToData(angle);
//        qDebug("-> v: %f, n: %f", -angle, convertDataToAngle(convertAngleToData(-angle)));

        //calc dist from cell n to cell n-1
        qreal len = (scanCell->getRelPosition() - scanCellPre1->getRelPosition()).length();
        tokenMem[Enums::Scanner::OUT_DISTANCE] = CodingPhysicalQuantities::convertShiftLenToData(len);
    }

    //scan cell
    quint32 e = qFloor(scanCell->getEnergy());
    if( e > 255 )        //restrict value to 8 bit
        e = 255;
    tokenMem[Enums::Scanner::OUT_ENERGY] = e;
    tokenMem[Enums::Scanner::OUT_CELL_MAX_CONNECTIONS] = scanCell->getMaxConnections();
    tokenMem[Enums::Scanner::OUT_CELL_BRANCH_NO] = scanCell->getBranchNumber();
	auto metadata = scanCell->getMetadata();
	tokenMem[Enums::Scanner::OUT_CELL_METADATA] = metadata.color;
	CellFunction* scanCellFunction = scanCell->getFeatures()->findObject<CellFunction>();
    tokenMem[Enums::Scanner::OUT_CELL_FUNCTION] = static_cast<quint8>(scanCellFunction->getType());
    QByteArray data = scanCellFunction->getInternalData();
	tokenMem.replace(Enums::Scanner::OUT_CELL_FUNCTION_DATA, data.size(), data);
	tokenMem.left(_context->getSimulationParameters()->tokenMemorySize);

    //scan cluster
    quint32 mass = qFloor(cell->getCluster()->getMass());
    if( mass > 255 )        //restrict value to 8 bit
        mass = 255;
    tokenMem[Enums::Scanner::OUT_MASS] = mass;
    return processingResult;
}

