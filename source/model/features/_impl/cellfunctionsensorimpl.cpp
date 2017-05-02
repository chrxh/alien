#include <QtCore/qmath.h>

#include "model/modelsettings.h"
#include "model/context/unitcontext.h"
#include "model/context/simulationparameters.h"
#include "model/context/cellmap.h"
#include "model/context/spacemetric.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/physics/physics.h"
#include "model/physics/codingphysicalquantities.h"

#include "cellfunctionsensorimpl.h"

CellFunctionSensorImpl::CellFunctionSensorImpl (UnitContext* context)
    : CellFunction(context)
{
}

CellFeature::ProcessingResult CellFunctionSensorImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    CellCluster* cluster(cell->getCluster());
	auto& tokenMem = token->getMemoryRef();
	quint8 cmd = tokenMem[Enums::Sensor::IN] % 5;
	auto cellMap = _context->getCellMap();
	auto metric = _context->getTopology();
	auto parameters = _context->getSimulationParameters();

    if( cmd == Enums::SensorIn::DO_NOTHING ) {
        tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::NOTHING_FOUND;
        return processingResult;
    }
    quint8 minMass = tokenMem[Enums::Sensor::IN_MIN_MASS];
    quint8 maxMass = tokenMem[Enums::Sensor::IN_MAX_MASS];
    qreal minMassReal = static_cast<qreal>(minMass);
    qreal maxMassReal = static_cast<qreal>(maxMass);
    if( maxMass == 0 )
        maxMassReal = 16000;    //large value => no max mass check

    //scanning vicinity?
    if( cmd == Enums::SensorIn::SEARCH_VICINITY ) {
        QVector3D cellPos = cell->calcPosition(_context);
//        auto time1 = high_resolution_clock::now();
        CellCluster* otherCluster = cellMap->getNearbyClusterFast(cellPos, parameters->cellFunctionSensorRange
            , minMassReal, maxMassReal, cluster);
//        nanoseconds diff1 = high_resolution_clock::now()- time1;
//        cout << "Dauer: " << diff1.count() << endl;
        if( otherCluster ) {
            tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::CLUSTER_FOUND;
            tokenMem[Enums::Sensor::OUT_MASS] = CodingPhysicalQuantities::convertURealToData(otherCluster->getMass());

            //calc relative angle
            QVector3D dir  = metric->displacement(cell->calcPosition(), otherCluster->getPosition()).normalized();
            qreal cellOrientationAngle = Physics::angleOfVector(-cell->getRelPosition() + previousCell->getRelPosition());
            qreal relAngle = Physics::angleOfVector(dir) - cellOrientationAngle - cluster->getAngle();
            tokenMem[Enums::Sensor::INOUT_ANGLE] = CodingPhysicalQuantities::convertAngleToData(relAngle);

            //calc distance by scanning along beam
            QVector3D beamPos = cell->calcPosition(true);
            QVector3D scanPos;
            for(int d = 1; d < parameters->cellFunctionSensorRange; d += 2) {
                beamPos += 2.0*dir;
                for(int rx = -1; rx < 2; ++rx)
                    for(int ry = -1; ry < 2; ++ry) {
                        scanPos = beamPos;
                        scanPos.setX(scanPos.x()+rx);
                        scanPos.setY(scanPos.y()+ry);
                        Cell* scanCell = cellMap->getCell(scanPos);
                        if( scanCell ) {
                            if( scanCell->getCluster() == otherCluster ) {
                                qreal dist = metric->displacement(scanCell->calcPosition(), cell->calcPosition()).length();
                                tokenMem[Enums::Sensor::OUT_DISTANCE] = CodingPhysicalQuantities::convertURealToData(dist);
                                return processingResult;
                            }
                        }
                    }
            }
            tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::NOTHING_FOUND;
        }
        else
            tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::NOTHING_FOUND;
        return processingResult;
    }

    //scanning in a particular direction?
    QVector3D cellRelPos(cluster->calcPosition(cell)-cluster->getPosition());
    QVector3D dir(0.0, 0.0, 0.0);
    if( cmd == Enums::SensorIn::SEARCH_BY_ANGLE ) {
        qreal relAngle = CodingPhysicalQuantities::convertDataToAngle(tokenMem[Enums::Sensor::INOUT_ANGLE]);
        qreal angle = Physics::angleOfVector(-cell->getRelPosition() + previousCell->getRelPosition()) + cluster->getAngle() + relAngle;
        dir = Physics::unitVectorOfAngle(angle);
    }
    if( cmd == Enums::SensorIn::SEARCH_FROM_CENTER ) {
        dir = cellRelPos.normalized();
    }
    if( cmd == Enums::SensorIn::SEARCH_TOWARD_CENTER ) {
        dir = -cellRelPos.normalized();
    }

    //scan along beam
    QList< Cell* > hitListCell;
    QVector3D beamPos = cell->calcPosition(true);
    QVector3D scanPos;
    for(int d = 1; d < parameters->cellFunctionSensorRange; d += 2) {
        beamPos += 2.0*dir;
        for(int rx = -1; rx < 2; ++rx)
            for(int ry = -1; ry < 2; ++ry) {
                scanPos = beamPos;
                scanPos.setX(scanPos.x()+rx);
                scanPos.setY(scanPos.y()+ry);
                Cell* scanCell = cellMap->getCell(scanPos);
                if( scanCell ) {
                    if( scanCell->getCluster() != cluster ) {

                        //scan masses
                        qreal mass = scanCell->getCluster()->getMass();
                        if( mass >= (minMassReal-ALIEN_PRECISION) && mass <= (maxMassReal+ALIEN_PRECISION) )
                            hitListCell << scanCell;
                    }
                }
            }

        //strike?
        if( !hitListCell.isEmpty() ) {

            //find largest cluster
            qreal largestMass = 0.0;
            Cell* largestClusterCell = 0;
            foreach( Cell* hitCell, hitListCell) {
                if( hitCell->getCluster()->getMass() > largestMass ) {
                    largestMass = hitCell->getCluster()->getMass();
                    largestClusterCell = hitCell;
                }
            }
            tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::CLUSTER_FOUND;
            qreal dist = metric->displacement(largestClusterCell->calcPosition(), cell->calcPosition()).length();
            tokenMem[Enums::Sensor::OUT_DISTANCE] = CodingPhysicalQuantities::convertURealToData(dist);
            tokenMem[Enums::Sensor::OUT_MASS] = CodingPhysicalQuantities::convertURealToData(largestClusterCell->getCluster()->getMass());
//            tokenMem[static_cast<int>(SENSOR::INOUT_ANGLE)] = convertURealToData(relAngle);
            return processingResult;
        }
    }
    tokenMem[Enums::Sensor::OUT] = Enums::SensorOut::NOTHING_FOUND;
    return processingResult;
}
