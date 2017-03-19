#include <qmath.h>
#include <QString>
#include <QList>
#include <QtAlgorithms>
#include <QMatrix4x4>

#include "global/servicelocator.h"
#include "model/alienfacade.h"
#include "model/entities/entityfactory.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/physics/physics.h"
#include "model/physics/codingphysicalquantities.h"
#include "model/config.h"
#include "model/simulationcontext.h"
#include "model/cellmap.h"
#include "model/topology.h"
#include "model/simulationparameters.h"

#include "cellfunctionconstructorimpl.h"

using ACTIVATE_TOKEN = Cell::ActivateToken;
using UPDATE_TOKEN_ACCESS_NUMBER = Cell::UpdateTokenAccessNumber;

CellFunctionConstructorImpl::CellFunctionConstructorImpl (SimulationContext* context)
    : CellFunction(context)
    , _cellMap(context->getCellMap())
    , _topology(context->getTopology())
	, _parameters(context->getSimulationParameters())
{
}

namespace {
    Enums::CellFunction::Type convertCellTypeNumberToName (int type)
    {
        type = type % Enums::CellFunction::_COUNTER;
        return static_cast< Enums::CellFunction::Type >(type);
    }

    Cell* constructNewCell (Cell* baseCell, QVector3D posOfNewCell, int maxConnections
        , int tokenAccessNumber, int cellType, QByteArray cellFunctionData, SimulationContext* context)
    {
        AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
        Cell* newCell = facade->buildFeaturedCell(context->getSimulationParameters()->NEW_CELL_ENERGY
			, convertCellTypeNumberToName(cellType), cellFunctionData, context);
        CellCluster* cluster = baseCell->getCluster();
        newCell->setMaxConnections(maxConnections);
        newCell->setTokenBlocked(true);
        newCell->setTokenAccessNumber(tokenAccessNumber);
		CellMetadata meta;
		meta.color = baseCell->getMetadata().color;
        newCell->setMetadata(meta);
        cluster->addCell(newCell, posOfNewCell);
        return newCell;
    }

    Cell* obstacleCheck (CellCluster* cluster, bool safeMode, CellMap* cellMap, Topology* topology, SimulationParameters* parameters)
    {
        foreach( Cell* cell, cluster->getCellsRef() ) {
            QVector3D pos = cluster->calcPosition(cell, true);

            for(int dx = -1; dx < 2; ++dx ) {
                for(int dy = -1; dy < 2; ++dy ) {
                    Cell* obstacleCell = cellMap->getCell(pos+QVector3D(dx,dy,0.0));

                    //obstacle found?
                    if( obstacleCell ) {
                        if( topology->displacement(obstacleCell->getCluster()->calcPosition(obstacleCell), pos).length() <  parameters->CRIT_CELL_DIST_MIN ) {
                            if( safeMode ) {
                                if( obstacleCell != cell ) {
                                    cluster->clearCellsFromMap();
                                    return obstacleCell;
                                }
                            }
                            else {
                                if( obstacleCell->getCluster() != cluster ) {
                                    return obstacleCell;
                                }
                            }
                        }

                        //check also connected cells
                        for(int i = 0; i < obstacleCell->getNumConnections(); ++i) {
                            Cell* connectedObstacleCell = obstacleCell->getConnection(i);
                            if( topology->displacement(connectedObstacleCell->getCluster()->calcPosition(connectedObstacleCell), pos).length() < parameters->CRIT_CELL_DIST_MIN ) {
                                if( safeMode ) {
                                    if( connectedObstacleCell != cell ) {
                                        cluster->clearCellsFromMap();
                                        return connectedObstacleCell;
                                    }
                                }
                                else {
                                    if( connectedObstacleCell->getCluster() != cluster ) {
                                        return connectedObstacleCell;
                                    }
                                }
                            }
                        }

                    }
                }
            }
            if( safeMode )
                cellMap->setCell(pos, cell);
        }
        if( safeMode )
            cluster->clearCellsFromMap();

        //no obstacle
        return 0;
    }

    qreal averageEnergy (qreal e1, qreal e2)
    {
        return (e1 + e2)/2.0;
    }

    void separateConstruction (Cell* constructedCell, Cell* constructorCell, bool reduceConnection)
    {
        constructorCell->delConnection(constructedCell);
        if( reduceConnection ) {
           constructedCell->setMaxConnections(constructedCell->getMaxConnections()-1);
           constructorCell->setMaxConnections(constructorCell->getMaxConnections()-1);
        }
    }
}

CellFeature::ProcessingResult CellFunctionConstructorImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    CellCluster* cluster(cell->getCluster());
	auto& tokenMem = token->getMemoryRef();
	quint8 cmd = tokenMem[static_cast<int>(Enums::Constr::IN)] % 4;
    quint8 opt = tokenMem[static_cast<int>(Enums::Constr::IN_OPTION)] % 7;

    //do nothing?
    if( cmd == static_cast<int>(Enums::ConstrIn::DO_NOTHING) )
        return processingResult;

    //read shift length for construction site from token data
    qreal len = CodingPhysicalQuantities::convertDataToShiftLen(tokenMem[static_cast<int>(Enums::Constr::IN_DIST)]);
    if( len > _parameters->CRIT_CELL_DIST_MAX ) {        //length to large?
        tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_DIST);
        return processingResult;
    }

    //looking for construction site
    int numCon(cell->getNumConnections());
    Cell* constructionCell(0);
    for(int i = 0; i < numCon; ++i) {
        if( cell->getConnection(i)->isTokenBlocked() ) {
            constructionCell = cell->getConnection(i);
        }
    }

    //save relative position of cells
    QList< QVector3D > relPosCells;
    foreach( Cell* otherCell, cluster->getCellsRef() )
        relPosCells << otherCell->getRelPos();

    //construction already in progress?
    if( constructionCell ) {

        //determine construction site via connected components
        cell->delConnection(constructionCell);
        QList< Cell* > constructionSite;
        cluster->getConnectedComponent(constructionCell, constructionSite);

        //construction site only connected with "cell"?
        if( !constructionSite.contains(cell) ) {

            //identify constructor (remaining cells)
            QList< Cell* > constructor;
            cluster->getConnectedComponent(cell, constructor);

            //calc possible angle for rotation of the construction site
            qreal minAngleConstrSite = 360.0;
            foreach( Cell* otherCell, constructionSite ) {
                qreal r = (otherCell->getRelPos() - constructionCell->getRelPos()).length();
                if( _parameters->CRIT_CELL_DIST_MAX < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(_parameters->CRIT_CELL_DIST_MAX/(2.0*r))*radToDeg);
                    if( a < minAngleConstrSite )
                        minAngleConstrSite = a;
                }
            }
            qreal minAngleConstructor = 360.0;
            foreach( Cell* otherCell, constructor ) {
                qreal r = (otherCell->getRelPos() - constructionCell->getRelPos()).length();
                if( _parameters->CRIT_CELL_DIST_MAX < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(_parameters->CRIT_CELL_DIST_MAX/(2.0*r))*radToDeg);
                    if( a < minAngleConstructor )
                        minAngleConstructor = a;
                }
            }

            //read desired rotation angle from token
            qreal angleSum = CodingPhysicalQuantities::convertDataToAngle(tokenMem[static_cast<int>(Enums::Constr::INOUT_ANGLE)]);

            //calc angular masses with respect to "constructionCell"
            qreal angMassConstrSite = 0.0;
            qreal angMassConstructor = 0.0;
            foreach( Cell* otherCell, constructionSite )
                angMassConstrSite = angMassConstrSite + (otherCell->getRelPos() - constructionCell->getRelPos()).lengthSquared();
            foreach( Cell* otherCell, constructor )
                angMassConstructor = angMassConstructor + (otherCell->getRelPos() - constructionCell->getRelPos()).lengthSquared();

            //calc angles for construction site and constructor
            qreal angleConstrSite = angMassConstructor*angleSum/(angMassConstrSite + angMassConstructor);
            qreal angleConstructor = angMassConstrSite*angleSum/(angMassConstrSite + angMassConstructor);//angleSum - angleConstrSite;
            bool performRotationOnly = false;
            if( qAbs(angleConstrSite) > minAngleConstrSite ) {
                performRotationOnly = true;
                if( angleConstrSite >= 0.0 )
                    angleConstrSite = qAbs(minAngleConstrSite);
                if( angleConstrSite < 0.0 )
                    angleConstrSite = -qAbs(minAngleConstrSite);
            }
            if( qAbs(angleConstructor) > minAngleConstructor ) {
                performRotationOnly = true;
                if( angleConstructor >= 0.0 )
                    angleConstructor = qAbs(minAngleConstructor);
                if( angleConstructor < 0.0 )
                    angleConstructor = -qAbs(minAngleConstructor);
            }

            //round angle when restricted
            qreal discrErrorRotation = 0.0;
            if( performRotationOnly ) {
                angleConstrSite = angleConstrSite + discrErrorRotation/2.0;
                angleConstructor = angleConstructor + discrErrorRotation/2.0;
            }


            //calc rigid tranformation for construction site and constructor
            QMatrix4x4 transformConstrSite;
            transformConstrSite.setToIdentity();
            transformConstrSite.translate(constructionCell->getRelPos());
            transformConstrSite.rotate(angleConstrSite, 0.0, 0.0, 1.0);
            transformConstrSite.translate(-constructionCell->getRelPos());

            QMatrix4x4 transformConstructor;
            transformConstructor.setToIdentity();
            transformConstructor.translate(constructionCell->getRelPos());
            transformConstructor.rotate(-angleConstructor, 0.0, 0.0, 1.0);
            transformConstructor.translate(-constructionCell->getRelPos());

            //apply rigid transformation to construction site and constructor
            cluster->clearCellsFromMap();
            foreach( Cell* otherCell, constructionSite )
                otherCell->setRelPos(transformConstrSite.map(otherCell->getRelPos()));
            foreach( Cell* otherCell, constructor )
                otherCell->setRelPos(transformConstructor.map(otherCell->getRelPos()));

            //only rotation?
            if( performRotationOnly ) {

                //restore connection to "cell"
                cell->newConnection(constructionCell);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithoutUpdate();
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/_parameters->INTERNAL_TO_KINETIC_ENERGY;

                //not enough energy?
                if( token->getEnergy() <= (_parameters->NEW_CELL_ENERGY + eDiff + _parameters->MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_NO_ENERGY);

                    //restore cluster
                    foreach( Cell* otherCell, cluster->getCellsRef() ) {
                        otherCell->setRelPos(relPosCells.first());
                        relPosCells.removeFirst();
                    }
                    cluster->drawCellsToMap();
                    return processingResult;
                }

                //update relative coordinates
                cluster->updateRelCoordinates(true);

                //obstacle found?
                if( cmd != static_cast<int>(Enums::ConstrIn::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(Enums::ConstrIn::SAFE));
                    if( obstacleCheck(cluster, safeMode, _cellMap, _topology, _parameters) ) {
                        tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_OBSTACLE);

                        //restore construction site
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPos(relPosCells.first());
                            relPosCells.removeFirst();
                        }
                        cluster->drawCellsToMap();
                        return processingResult;
                    }
                }

                //update remaining cluster data
                cluster->updateAngularMass();
                cluster->setAngularVel(angularVelNew);

                //update token data
                tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::SUCCESS_ROT);
                tokenMem[static_cast<int>(Enums::Constr::INOUT_ANGLE)] = CodingPhysicalQuantities::convertAngleToData(angleSum - angleConstrSite - angleConstructor);
                cluster->drawCellsToMap();
            }

            //not only rotation but also creating new cell
            else {

                //calc translation vector for construction site
                QVector3D transOld = constructionCell->getRelPos() - cell->getRelPos();
                QVector3D trans = transOld.normalized() * len;
                QVector3D transFinish(0.0, 0.0, 0.0);
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) )
                    transFinish = trans;


                //shift construction site
                foreach( Cell* otherCell, constructionSite )
                    otherCell->setRelPos(otherCell->getRelPos() + trans + transFinish);

                //calc position for new cell
                QVector3D pos = cluster->relToAbsPos(cell->getRelPos() + transOld + transFinish);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/_parameters->INTERNAL_TO_KINETIC_ENERGY;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ))
                    tokenEnergy = _parameters->NEW_TOKEN_ENERGY;

                //not enough energy?
                if( token->getEnergy() <= (_parameters->NEW_CELL_ENERGY + tokenEnergy + eDiff + _parameters->MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_NO_ENERGY);

                    //restore construction site
                    foreach( Cell* otherCell, cluster->getCellsRef() ) {
                        otherCell->setRelPos(relPosCells.first());
                        relPosCells.removeFirst();
                    }
                    cluster->drawCellsToMap();

                    //restore connection from construction site to "cell"
                    cell->newConnection(constructionCell);
                    return processingResult;
                }

                //construct new cell
                quint8 maxCon = tokenMem[static_cast<int>(Enums::Constr::IN_CELL_MAX_CONNECTIONS)];
                if( maxCon < 2 )
                    maxCon = 2;
                if( maxCon > _parameters->MAX_CELL_CONNECTIONS )
                    maxCon = _parameters->MAX_CELL_CONNECTIONS;
                int tokenAccessNumber = tokenMem[static_cast<int>(Enums::Constr::IN_CELL_BRANCH_NO)] % _parameters->MAX_TOKEN_ACCESS_NUMBERS;
                Cell* newCell = constructNewCell(cell, pos, maxCon, tokenAccessNumber
					, tokenMem[static_cast<int>(Enums::Constr::IN_CELL_FUNCTION)]
					, tokenMem.mid(static_cast<int>(Enums::Constr::IN_CELL_FUNCTION_DATA))
					, _context);

                //obstacle found?
                if( cmd != static_cast<int>(Enums::ConstrIn::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(Enums::ConstrIn::SAFE));
                    if( obstacleCheck(cluster, safeMode, _cellMap, _topology, _parameters) ) {
                        tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_OBSTACLE);

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPos(relPosCells.first());
                            relPosCells.removeFirst();
                        }
                        cluster->updateAngularMass();
                        cluster->drawCellsToMap();

                        //restore connection from construction site to "cell"
                        cell->newConnection(constructionCell);
                        return processingResult;
                    }
                }

                //establish connections
                newCell->newConnection(cell);
                newCell->newConnection(constructionCell);

                //connect cell with construction site
                foreach(Cell* otherCell, constructionSite) {
                    if( (otherCell->getNumConnections() < _parameters->MAX_CELL_CONNECTIONS)
                            && (newCell->getNumConnections() < _parameters->MAX_CELL_CONNECTIONS)
                            && (otherCell !=constructionCell ) ) {
                        if (_topology->displacement(newCell->getRelPos(), otherCell->getRelPos()).length() <= (_parameters->CRIT_CELL_DIST_MAX + ALIEN_PRECISION) ) {

                            //CONSTR_IN_CELL_MAX_CONNECTIONS = 0 => set "maxConnections" automatically
                            if( tokenMem.at(static_cast<int>(Enums::Constr::IN_CELL_MAX_CONNECTIONS)) == 0 ) {
                                if( newCell->getNumConnections() == newCell->getMaxConnections() ) {
                                    newCell->setMaxConnections(newCell->getMaxConnections()+1);
                                }
                                if( otherCell->getNumConnections() == otherCell->getMaxConnections() ) {
                                    otherCell->setMaxConnections(otherCell->getMaxConnections()+1);
                                }
                                newCell->newConnection(otherCell);
                            }
                            else {
                                if( (newCell->getNumConnections() < newCell->getMaxConnections() )
                                    && (otherCell->getNumConnections() < otherCell->getMaxConnections()) )
                                    newCell->newConnection(otherCell);
                            }
                        }
                    }
                }

                //update token energy
                token->setEnergy(token->getEnergy() - newCell->getEnergy() - eDiff);

                //average cell energy if token is created
                if( (opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    qreal av = averageEnergy(cell->getEnergy(), newCell->getEnergy());
                    cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //allow token access to old "construction cell"
                constructionCell->setTokenBlocked(false);

                //finish construction site?
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_NO_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) )
                    newCell->setTokenBlocked(false);

                //separate construction site?
                if( opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, false);
                }
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, true);
                }

                //update token data
                tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::SUCCESS);
                tokenMem[static_cast<int>(Enums::Constr::INOUT_ANGLE)] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
                if( (opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    if( newCell->getNumToken(true) < _parameters->CELL_TOKENSTACKSIZE ) {
						auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
						auto token = factory->buildToken(_context, _parameters->NEW_TOKEN_ENERGY);
                        newCell->addToken(token, ACTIVATE_TOKEN::LATER, UPDATE_TOKEN_ACCESS_NUMBER::YES);
                        token->setEnergy(token->getEnergy() - _parameters->NEW_TOKEN_ENERGY);
                    }
                }
                if( opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN) ) {
                    if( newCell->getNumToken(true) < _parameters->CELL_TOKENSTACKSIZE ) {
                        auto dup = token->duplicate();
                        dup->setEnergy(_parameters->NEW_TOKEN_ENERGY);
                        newCell->addToken(dup, ACTIVATE_TOKEN::LATER, UPDATE_TOKEN_ACCESS_NUMBER::YES);
                        token->setEnergy(token->getEnergy() - _parameters->NEW_TOKEN_ENERGY);
                    }
                }
            }
        }

        //construction site connected with other cells than "cell"? => error
        else {
            tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_CONNECTION);

            //restore connection to "cell"
            cell->newConnection(constructionCell);
        }
    }

    //start new construction site?
    else {

        //new cell connection possible?
        if( numCon < _parameters->MAX_CELL_CONNECTIONS ) {

            //is there any connection?
            if( numCon > 0 ) {

                //find biggest angle gap for new cell
                QVector< qreal > angles(numCon);
                for(int i = 0; i < numCon; ++i) {
                    QVector3D displacement = cluster->calcPosition(cell->getConnection(i),true)-cluster->calcPosition(cell, true);
                    _topology->correctDisplacement(displacement);
                    angles[i] = Physics::angleOfVector(displacement);
                }
                qSort(angles);
                qreal largestAnglesDiff = 0.0;
                qreal angleGap = 0.0;
                for(int i = 0; i < numCon; ++i) {
                    qreal angleDiff = angles[(i+1)%numCon]-angles[i];
                    if( angleDiff <= 0.0 )
                        angleDiff += 360.0;
                    if( angleDiff > 360.0  )
                        angleDiff -= 360.0;
                    if( angleDiff > largestAnglesDiff ) {
                        largestAnglesDiff = angleDiff;
                        angleGap = angles[i] + angleDiff/2.0;
                    }
                }

                //calc start angle
                angleGap = angleGap + CodingPhysicalQuantities::convertDataToAngle(tokenMem[static_cast<int>(Enums::Constr::INOUT_ANGLE)]);

                //calc coordinates for new cell from angle gap and construct cell
                QVector3D angleGapPos = Physics::unitVectorOfAngle(angleGap)*_parameters->CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
                QVector3D pos = cluster->calcPosition(cell)+angleGapPos;
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) )
                    pos = pos + angleGapPos;


                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/_parameters->INTERNAL_TO_KINETIC_ENERGY;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN) )
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) )
                    tokenEnergy = _parameters->NEW_TOKEN_ENERGY;

                //not enough energy?
                if( token->getEnergy() <= (_parameters->NEW_CELL_ENERGY + tokenEnergy + eDiff + _parameters->MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_NO_ENERGY);
                    return processingResult;
                }

                //construct new cell
                cluster->clearCellsFromMap();
                quint8 maxCon = tokenMem[static_cast<int>(Enums::Constr::IN_CELL_MAX_CONNECTIONS)];
                if( maxCon < 1 )
                    maxCon = 1;
                if( maxCon > _parameters->MAX_CELL_CONNECTIONS )
                    maxCon = _parameters->MAX_CELL_CONNECTIONS;
                int tokenAccessNumber = tokenMem[static_cast<int>(Enums::Constr::IN_CELL_BRANCH_NO)]
                        % _parameters->MAX_TOKEN_ACCESS_NUMBERS;
                Cell* newCell = constructNewCell(cell, pos, maxCon, tokenAccessNumber
					, tokenMem[static_cast<int>(Enums::Constr::IN_CELL_FUNCTION)]
					, tokenMem.mid(static_cast<int>(Enums::Constr::IN_CELL_FUNCTION_DATA)), _context);

                //obstacle found?
                if( cmd != static_cast<int>(Enums::ConstrIn::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(Enums::ConstrIn::SAFE));
                    if( obstacleCheck(cluster, safeMode, _cellMap, _topology, _parameters) ) {
                        tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_OBSTACLE);

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPos(relPosCells.first());
                            relPosCells.removeFirst();
                        }
                        cluster->drawCellsToMap();
                        cluster->updateAngularMass();
                        return processingResult;
                    }
                }

                //establish connection
                if( cell->getNumConnections() == cell->getMaxConnections() )
                    cell->setMaxConnections(cell->getMaxConnections()+1);
                cell->newConnection(newCell);

                //update token energy
                token->setEnergy(token->getEnergy() - newCell->getEnergy() - eDiff);

                //average cell energy if token is created
                if( (opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    qreal av = averageEnergy(cell->getEnergy(), newCell->getEnergy());
                    cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //finish construction site?
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_NO_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) )
                    newCell->setTokenBlocked(false);

                //separate construction site?
                if( opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, false);
                }
                if( (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, true);
                }

                //update token data
                tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::SUCCESS);
                tokenMem[static_cast<int>(Enums::Constr::INOUT_ANGLE)] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
				auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
				if ((opt == static_cast<int>(Enums::ConstrInOption::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)) ) {
					auto token = factory->buildToken(_context, _parameters->NEW_TOKEN_ENERGY);
                    newCell->addToken(token, ACTIVATE_TOKEN::LATER, UPDATE_TOKEN_ACCESS_NUMBER::YES);
                    token->setEnergy(token->getEnergy() - _parameters->NEW_TOKEN_ENERGY);
                }
                if( opt == static_cast<int>(Enums::ConstrInOption::CREATE_DUP_TOKEN) ) {
                    auto dup = token->duplicate();
                    dup->setEnergy(_parameters->NEW_TOKEN_ENERGY);
                    newCell->addToken(dup, ACTIVATE_TOKEN::LATER, UPDATE_TOKEN_ACCESS_NUMBER::YES);
                    token->setEnergy(token->getEnergy() - _parameters->NEW_TOKEN_ENERGY);
                }

            }

            //no connection by now
            else {
                tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_CONNECTION);
            }
        }

        //no new connection possible
        else {

            //error code
            tokenMem[static_cast<int>(Enums::Constr::OUT)] = static_cast<int>(Enums::ConstrOut::ERROR_CONNECTION);
        }
    }
    return processingResult;
}

