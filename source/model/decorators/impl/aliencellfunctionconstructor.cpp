#include "aliencellfunctionconstructor.h"
#include "model/decorators/aliencelldecoratorfactory.h"
#include "model/entities/entityfactory.h"
#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"
#include "model/physics/physics.h"
#include "model/simulationsettings.h"

#include "global/servicelocator.h"

#include <qmath.h>
#include <QString>
#include <QList>
#include <QtAlgorithms>



AlienCellFunctionConstructor::AlienCellFunctionConstructor (AlienCell* cell, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{
}

AlienCellFunctionConstructor::AlienCellFunctionConstructor (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

AlienCellFunctionConstructor::AlienCellFunctionConstructor (AlienCell* cell, QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

namespace {
    CellFunctionType convertCellTypeNumberToName (int type)
    {
        type = type % static_cast< int >(CellFunctionType::_COUNTER);
        return static_cast< CellFunctionType >(type);
    }

    AlienCell* constructNewCell (AlienCell* baseCell, QVector3D posOfNewCell, int maxConnections
        , int tokenAccessNumber, int cellType, quint8* cellFunctionData, AlienGrid*& grid)
    {
        EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
        AlienCell* newCell = entityFactory->buildCell(simulationParameters.NEW_CELL_ENERGY, grid);
        AlienCellCluster* cluster = baseCell->getCluster();
        newCell->setMaxConnections(maxConnections);
        newCell->setTokenBlocked(true);
        newCell->setTokenAccessNumber(tokenAccessNumber);
        AlienCellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<AlienCellDecoratorFactory>();
        decoratorFactory->addCellFunction(newCell, convertCellTypeNumberToName(cellType), cellFunctionData, grid);
        decoratorFactory->addEnergyGuidance(newCell, grid);
        newCell->setColor(baseCell->getColor());
        cluster->addCell(newCell, posOfNewCell);
        return newCell;
    }

    AlienCell* obstacleCheck (AlienCellCluster* cluster, bool safeMode, AlienGrid*& grid)
    {
        //obstacle check
        foreach( AlienCell* cell, cluster->getCells() ) {
            QVector3D pos = cluster->calcPosition(cell, true);

            for(int dx = -1; dx < 2; ++dx ) {
                for(int dy = -1; dy < 2; ++dy ) {
                    AlienCell* obstacleCell = grid->getCell(pos+QVector3D(dx,dy,0.0));

                    //obstacle found?
                    if( obstacleCell ) {
                        if( grid->displacement(obstacleCell->getCluster()->calcPosition(obstacleCell), pos).length() < simulationParameters.CRIT_CELL_DIST_MIN ) {
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
                            AlienCell* connectedObstacleCell = obstacleCell->getConnection(i);
                            if( grid->displacement(connectedObstacleCell->getCluster()->calcPosition(connectedObstacleCell), pos).length() < simulationParameters.CRIT_CELL_DIST_MIN ) {
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
                grid->setCell(pos, cell);
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

    void separateConstruction (AlienCell* constructedCell, AlienCell* constructorCell, bool reduceConnection)
    {
        constructorCell->delConnection(constructedCell);
        if( reduceConnection ) {
           constructedCell->setMaxConnections(constructedCell->getMaxConnections()-1);
           constructorCell->setMaxConnections(constructorCell->getMaxConnections()-1);
        }
    }
}

AlienCellDecorator::ProcessingResult AlienCellFunctionConstructor::process (AlienToken* token, AlienCell* previousCell)
{
    AlienCell::ProcessingResult processingResult = _cell->process(token, previousCell);
    AlienCellCluster* cluster(_cell->getCluster());
    quint8 cmd = token->memory[static_cast<int>(CONSTR::IN)] % 4;
    quint8 opt = token->memory[static_cast<int>(CONSTR::IN_OPTION)] % 7;

    //do nothing?
    if( cmd == static_cast<int>(CONSTR_IN::DO_NOTHING) )
        return processingResult;

    //read shift length for construction site from token data
    qreal len = convertDataToShiftLen(token->memory[static_cast<int>(CONSTR::IN_DIST)]);
    if( len > simulationParameters.CRIT_CELL_DIST_MAX ) {        //length to large?
        token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_DIST);
        return processingResult;
    }

    //looking for construction site
    int numCon(_cell->getNumConnections());
    AlienCell* constructionCell(0);
    for(int i = 0; i < numCon; ++i) {
        if( _cell->getConnection(i)->isTokenBlocked() ) {
            constructionCell = _cell->getConnection(i);
        }
    }

    //save relative position of cells
    QList< QVector3D > relPosCells;
    foreach( AlienCell* otherCell, cluster->getCells() )
        relPosCells << otherCell->getRelPos();

    //construction already in progress?
    if( constructionCell ) {

        //determine construction site via connected components
        _cell->delConnection(constructionCell);
        QList< AlienCell* > constructionSite;
        cluster->getConnectedComponent(constructionCell, constructionSite);

        //construction site only connected with "cell"?
        if( !constructionSite.contains(_cell) ) {

            //identify constructor (remaining cells)
            QList< AlienCell* > constructor;
            cluster->getConnectedComponent(_cell, constructor);

            //calc possible angle for rotation of the construction site
            qreal minAngleConstrSite = 360.0;
            foreach( AlienCell* otherCell, constructionSite ) {
                qreal r = (otherCell->getRelPos() - constructionCell->getRelPos()).length();
                if( simulationParameters.CRIT_CELL_DIST_MAX < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(simulationParameters.CRIT_CELL_DIST_MAX/(2.0*r))*radToDeg);
                    if( a < minAngleConstrSite )
                        minAngleConstrSite = a;
                }
            }
            qreal minAngleConstructor = 360.0;
            foreach( AlienCell* otherCell, constructor ) {
                qreal r = (otherCell->getRelPos() - constructionCell->getRelPos()).length();
                if( simulationParameters.CRIT_CELL_DIST_MAX < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(simulationParameters.CRIT_CELL_DIST_MAX/(2.0*r))*radToDeg);
                    if( a < minAngleConstructor )
                        minAngleConstructor = a;
                }
            }

            //read desired rotation angle from token
            qreal angleSum = convertDataToAngle(token->memory[static_cast<int>(CONSTR::INOUT_ANGLE)]);

            //calc angular masses with respect to "constructionCell"
            qreal angMassConstrSite = 0.0;
            qreal angMassConstructor = 0.0;
            foreach( AlienCell* otherCell, constructionSite )
                angMassConstrSite = angMassConstrSite + (otherCell->getRelPos() - constructionCell->getRelPos()).lengthSquared();
            foreach( AlienCell* otherCell, constructor )
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
//                angleConstrSite = convertDataToAngle(convertAngleToData(angleConstrSite));
//                angleConstructor = convertDataToAngle(convertAngleToData(angleConstructor));
//                discrErrorRotation = angleSum - angleConstrSite - angleConstructor;
//                discrErrorRotation = discrErrorRotation -convertDataToAngle(convertAngleToData(discrErrorRotation));
                angleConstrSite = angleConstrSite + discrErrorRotation/2.0;
                angleConstructor = angleConstructor + discrErrorRotation/2.0;
            }


            //calc rigid tranformation for construction site and constructor
            QMatrix4x4 transformConstrSite;
            transformConstrSite.setToIdentity();
            transformConstrSite.translate(constructionCell->getRelPos());
            transformConstrSite.rotate(angleConstrSite, 0.0, 0.0, 1.0);
//            if( performRotationOnly )
                transformConstrSite.translate(-constructionCell->getRelPos());
//            else
//                transformConstrSite.translate(-cell->getRelPos());

            QMatrix4x4 transformConstructor;
            transformConstructor.setToIdentity();
            transformConstructor.translate(constructionCell->getRelPos());
            transformConstructor.rotate(-angleConstructor, 0.0, 0.0, 1.0);
            transformConstructor.translate(-constructionCell->getRelPos());

            //apply rigid transformation to construction site and constructor
            cluster->clearCellsFromMap();
            foreach( AlienCell* otherCell, constructionSite )
                otherCell->setRelPos(transformConstrSite.map(otherCell->getRelPos()));
            foreach( AlienCell* otherCell, constructor )
                otherCell->setRelPos(transformConstructor.map(otherCell->getRelPos()));

            //only rotation?
            if( performRotationOnly ) {

                //restore connection to "cell"
                _cell->newConnection(constructionCell);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithoutUpdate();
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;

                //not enough energy?
                if( token->energy <= (simulationParameters.NEW_CELL_ENERGY + eDiff + simulationParameters.MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_NO_ENERGY);

                    //restore cluster
                    foreach( AlienCell* otherCell, cluster->getCells() ) {
                        otherCell->setRelPos(relPosCells.first());
                        relPosCells.removeFirst();
                    }
//                    transform = transform.inverted();
//                    foreach( AlienCell* otherCell, constructionSite )
//                        otherCell->setRelPos(transform.map(otherCell->getRelPos()));
                    cluster->drawCellsToMap();
                    return processingResult;
                }

                //update relative coordinates
                cluster->updateRelCoordinates(true);

                //obstacle found?
                if( cmd != static_cast<int>(CONSTR_IN::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(CONSTR_IN::SAFE));
                    if( obstacleCheck(cluster, safeMode, _grid) ) {
                        token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_OBSTACLE);

                        //restore construction site
                        foreach( AlienCell* otherCell, cluster->getCells() ) {
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
                token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::SUCCESS_ROT);
                token->memory[static_cast<int>(CONSTR::INOUT_ANGLE)] = convertAngleToData(angleSum - angleConstrSite - angleConstructor);
                cluster->drawCellsToMap();
            }

            //not only rotation but also creating new cell
            else {

                //calc translation vector for construction site
                QVector3D transOld = constructionCell->getRelPos() - _cell->getRelPos();
                QVector3D trans = transOld.normalized() * len;
                QVector3D transFinish(0.0, 0.0, 0.0);
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) )
                    transFinish = trans;


                //shift construction site
                foreach( AlienCell* otherCell, constructionSite )
                    otherCell->setRelPos(otherCell->getRelPos() + trans + transFinish);

                //calc position for new cell
                QVector3D pos = cluster->relToAbsPos(_cell->getRelPos() + transOld + transFinish);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED) ))
                    tokenEnergy = simulationParameters.NEW_TOKEN_ENERGY;

                //not enough energy?
                if( token->energy <= (simulationParameters.NEW_CELL_ENERGY + tokenEnergy + eDiff + simulationParameters.MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_NO_ENERGY);

                    //restore construction site
                    foreach( AlienCell* otherCell, cluster->getCells() ) {
                        otherCell->setRelPos(relPosCells.first());
                        relPosCells.removeFirst();
                    }
                    cluster->drawCellsToMap();
//                    transform = transform.inverted();
//                    foreach( AlienCell* otherCell, constructionSite )
//                        otherCell->setRelPos(transform.map(otherCell->getRelPos()));

                    //restore connection from construction site to "cell"
                    _cell->newConnection(constructionCell);
                    return processingResult;
                }

                //construct new cell
                quint8 maxCon = token->memory[static_cast<int>(CONSTR::IN_CELL_MAX_CONNECTIONS)];
                if( maxCon < 2 )
                    maxCon = 2;
                if( maxCon > simulationParameters.MAX_CELL_CONNECTIONS )
                    maxCon = simulationParameters.MAX_CELL_CONNECTIONS;
                int tokenAccessNumber = token->memory[static_cast<int>(CONSTR::IN_CELL_BRANCH_NO)] % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
                AlienCell* newCell = constructNewCell(_cell, pos, maxCon, tokenAccessNumber,
                                                      token->memory[static_cast<int>(CONSTR::IN_CELL_FUNCTION)],
                                                      &(token->memory[static_cast<int>(CONSTR::IN_CELL_FUNCTION_DATA)]), _grid);

                //obstacle found?
                if( cmd != static_cast<int>(CONSTR_IN::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(CONSTR_IN::SAFE));
                    if( obstacleCheck(cluster, safeMode, _grid) ) {
                        token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_OBSTACLE);

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( AlienCell* otherCell, cluster->getCells() ) {
                            otherCell->setRelPos(relPosCells.first());
                            relPosCells.removeFirst();
                        }
                        cluster->updateAngularMass();
                        cluster->drawCellsToMap();

                        //restore connection from construction site to "cell"
                        _cell->newConnection(constructionCell);
                        return processingResult;
                    }
                }

                //establish connections
                newCell->newConnection(_cell);
                newCell->newConnection(constructionCell);

                //connect cell with construction site
                foreach(AlienCell* otherCell, constructionSite) {
                    if( (otherCell->getNumConnections() < simulationParameters.MAX_CELL_CONNECTIONS)
                            && (newCell->getNumConnections() < simulationParameters.MAX_CELL_CONNECTIONS)
                            && (otherCell !=constructionCell ) ) {
                        if( _grid->displacement(newCell->getRelPos(), otherCell->getRelPos()).length() <= (simulationParameters.CRIT_CELL_DIST_MAX + ALIEN_PRECISION) ) {

                            //CONSTR_IN_CELL_MAX_CONNECTIONS = 0 => set "maxConnections" automatically
                            if( token->memory[static_cast<int>(CONSTR::IN_CELL_MAX_CONNECTIONS)] == 0 ) {
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
                token->energy = token->energy - newCell->getEnergy() - eDiff;

                //average cell energy if token is created
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    qreal av = averageEnergy(_cell->getEnergy(), newCell->getEnergy());
                    _cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //allow token access to old "construction cell"
                constructionCell->setTokenBlocked(false);

                //finish construction site?
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_NO_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) )
                    newCell->setTokenBlocked(false);

                //separate construction site?
                if( opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, _cell, false);
                }
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, _cell, true);
                }

                //update token data
                token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::SUCCESS);
                token->memory[static_cast<int>(CONSTR::INOUT_ANGLE)] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    if( newCell->getNumToken(true) < simulationParameters.CELL_TOKENSTACKSIZE ) {
                        newCell->addToken(new AlienToken(simulationParameters.NEW_TOKEN_ENERGY), false);
                        token->energy = token->energy - simulationParameters.NEW_TOKEN_ENERGY;
                    }
                }
                if( opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN) ) {
                    if( newCell->getNumToken(true) < simulationParameters.CELL_TOKENSTACKSIZE ) {
                        AlienToken* dup = token->duplicate();
                        dup->energy = simulationParameters.NEW_TOKEN_ENERGY;
                        newCell->addToken(dup, false);
                        token->energy = token->energy - simulationParameters.NEW_TOKEN_ENERGY;
                    }
                }
            }
        }

        //construction site connected with other cells than "cell"? => error
        else {
            token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_CONNECTION);

            //restore connection to "cell"
            _cell->newConnection(constructionCell);
        }
    }

    //start new construction site?
    else {

        //new cell connection possible?
        if( numCon < simulationParameters.MAX_CELL_CONNECTIONS ) {

            //is there any connection?
            if( numCon > 0 ) {

                //find biggest angle gap for new cell
                QVector< qreal > angles(numCon);
                for(int i = 0; i < numCon; ++i) {
                    QVector3D displacement = cluster->calcPosition(_cell->getConnection(i),true)-cluster->calcPosition(_cell, true);
                    _grid->correctDisplacement(displacement);
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
                angleGap = angleGap + convertDataToAngle(token->memory[static_cast<int>(CONSTR::INOUT_ANGLE)]);

                //calc coordinates for new cell from angle gap and construct cell
                QVector3D angleGapPos = Physics::unitVectorOfAngle(angleGap)*simulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
                QVector3D pos = cluster->calcPosition(_cell)+angleGapPos;
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) )
                    pos = pos + angleGapPos;


                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVel(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN) )
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) )
                    tokenEnergy = simulationParameters.NEW_TOKEN_ENERGY;

                //not enough energy?
                if( token->energy <= (simulationParameters.NEW_CELL_ENERGY + tokenEnergy + eDiff + simulationParameters.MIN_TOKEN_ENERGY + ALIEN_PRECISION) ) {
                    token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_NO_ENERGY);
                    return processingResult;
                }

                //construct new cell
                cluster->clearCellsFromMap();
                quint8 maxCon = token->memory[static_cast<int>(CONSTR::IN_CELL_MAX_CONNECTIONS)];
                if( maxCon < 1 )
                    maxCon = 1;
                if( maxCon > simulationParameters.MAX_CELL_CONNECTIONS )
                    maxCon = simulationParameters.MAX_CELL_CONNECTIONS;
                int tokenAccessNumber = token->memory[static_cast<int>(CONSTR::IN_CELL_BRANCH_NO)]
                        % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
                AlienCell* newCell = constructNewCell(_cell, pos, maxCon, tokenAccessNumber,
                                                      token->memory[static_cast<int>(CONSTR::IN_CELL_FUNCTION)],
                                                      &(token->memory[static_cast<int>(CONSTR::IN_CELL_FUNCTION_DATA)]), _grid);

                //obstacle found?
                if( cmd != static_cast<int>(CONSTR_IN::BRUTEFORCE)) {
                    bool safeMode = (cmd == static_cast<int>(CONSTR_IN::SAFE));
                    if( obstacleCheck(cluster, safeMode, _grid) ) {
                        token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_OBSTACLE);

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( AlienCell* otherCell, cluster->getCells() ) {
                            otherCell->setRelPos(relPosCells.first());
                            relPosCells.removeFirst();
                        }
                        cluster->drawCellsToMap();
                        cluster->updateAngularMass();
                        return processingResult;
                    }
                }

                //establish connection
                if( _cell->getNumConnections() == _cell->getMaxConnections() )
                    _cell->setMaxConnections(_cell->getMaxConnections()+1);
                _cell->newConnection(newCell);

                //update token energy
                token->energy = token->energy - newCell->getEnergy() - eDiff;

                //average cell energy if token is created
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    qreal av = averageEnergy(_cell->getEnergy(), newCell->getEnergy());
                    _cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //finish construction site?
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_NO_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) )
                    newCell->setTokenBlocked(false);

                //separate construction site?
                if( opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, _cell, false);
                }
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, _cell, true);
                }

                //update token data
                token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::SUCCESS);
                token->memory[static_cast<int>(CONSTR::INOUT_ANGLE)] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
                if( (opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN))
                        || (opt == static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)) ) {
                    newCell->addToken(new AlienToken(simulationParameters.NEW_TOKEN_ENERGY), false);
                    token->energy = token->energy - simulationParameters.NEW_TOKEN_ENERGY;
                }
                if( opt == static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN) ) {
                    AlienToken* dup = token->duplicate();
                    dup->energy = simulationParameters.NEW_TOKEN_ENERGY;
                    newCell->addToken(dup, false);
                    token->energy = token->energy - simulationParameters.NEW_TOKEN_ENERGY;
                }

            }

            //no connection by now
            else {
                token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_CONNECTION);
            }
        }

        //no new connection possible
        else {

            //error code
            token->memory[static_cast<int>(CONSTR::OUT)] = static_cast<int>(CONSTR_OUT::ERROR_CONNECTION);
        }
    }
    return processingResult;
}

void AlienCellFunctionConstructor::serialize (QDataStream& stream)
{
    AlienCellDecorator::serialize(stream);
}

