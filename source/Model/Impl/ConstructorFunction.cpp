#include <qmath.h>
#include <QString>
#include <QList>
#include <QtAlgorithms>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Local/EntityFactory.h"
#include "Model/Local/Cell.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Token.h"
#include "Model/Api/Physics.h"
#include "Model/Local/PhysicalQuantityConverter.h"
#include "Model/Api/Settings.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/SpacePropertiesLocal.h"
#include "Model/Api/SimulationParameters.h"

#include "ConstructorFunction.h"

using ACTIVATE_TOKEN = Cell::ActivateToken;
using UPDATE_TOKEN_ACCESS_NUMBER = Cell::UpdateTokenBranchNumber;

ConstructorFunction::ConstructorFunction (UnitContext* context)
    : CellFunction(context)
{
}

namespace {
    Enums::CellFunction::Type convertCellTypeNumberToName (int type)
    {
        type = type % Enums::CellFunction::_COUNTER;
        return static_cast< Enums::CellFunction::Type >(type);
    }

    Cell* constructNewCell (Cell* baseCell, QVector2D posOfNewCell, int maxConnections
        , int tokenAccessNumber, quint8 metadata, int cellType, QByteArray cellFunctionData, UnitContext* context)
    {
		EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
		CellMetadata meta;
		meta.color = metadata;

		int length = static_cast<int>(static_cast<uint8_t>(cellFunctionData[0]));
		QByteArray constData = cellFunctionData.mid(1, length);

		auto desc = CellDescription().setEnergy(context->getSimulationParameters()->cellCreationEnergy).setMaxConnections(maxConnections)
			.setTokenBranchNumber(tokenAccessNumber).setFlagTokenBlocked(true).setMetadata(meta)
			.setCellFeature(CellFeatureDescription().setType(convertCellTypeNumberToName(cellType)).setConstData(constData))
			.setPos(posOfNewCell);
		auto cell = factory->build(desc, baseCell->getCluster(), context);
		baseCell->getCluster()->updateInternals();
		return cell;
    }

    Cell* obstacleCheck (Cluster* cluster, bool safeMode, CellMap* cellMap, SpacePropertiesLocal* metric, SimulationParameters* parameters)
    {
        foreach( Cell* cell, cluster->getCellsRef() ) {
            QVector2D pos = cluster->calcPosition(cell, true);

            for(int dx = -1; dx < 2; ++dx ) {
                for(int dy = -1; dy < 2; ++dy ) {
                    Cell* obstacleCell = cellMap->getCell(pos+QVector2D(dx, dy));

                    //obstacle found?
                    if( obstacleCell ) {
                        if( metric->displacement(obstacleCell->getCluster()->calcPosition(obstacleCell), pos).length() <  parameters->cellMinDistance ) {
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
                            if( metric->displacement(connectedObstacleCell->getCluster()->calcPosition(connectedObstacleCell), pos).length() < parameters->cellMinDistance ) {
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

	bool checkMaxRadiusOfCluster(Cluster* cluster, double lengthIncrement, SimulationParameters* parameters)
	{
		return cluster->getRadius() + lengthIncrement < parameters->clusterMaxRadius;
	}
}

CellFeatureChain::ProcessingResult ConstructorFunction::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    Cluster* cluster(cell->getCluster());
	auto& tokenMem = token->getMemoryRef();
	quint8 cmd = tokenMem[Enums::Constr::IN] % 4;
    quint8 opt = tokenMem[Enums::Constr::IN_OPTION] % 7;
	auto cellMap = _context->getCellMap();
	auto metric = _context->getSpaceProperties();
	auto parameters = _context->getSimulationParameters();

    //do nothing?
    if( cmd == Enums::ConstrIn::DO_NOTHING )
        return processingResult;

    //read shift length for construction site from token data
    qreal len = PhysicalQuantityConverter::convertDataToShiftLen(tokenMem[Enums::Constr::IN_DIST]);
    if( len > parameters->cellMaxDistance ) {        //length to large?
        tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_DIST;
        return processingResult;
    }

	if (!checkMaxRadiusOfCluster(cluster, len + 2, parameters)) {	//+2 because neighbor pixels are also retrieved
		tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_MAX_RADIUS;
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
    QList< QVector2D > relPosCells;
    foreach( Cell* otherCell, cluster->getCellsRef() )
        relPosCells << otherCell->getRelPosition();

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
                qreal r = (otherCell->getRelPosition() - constructionCell->getRelPosition()).length();
                if( parameters->cellMaxDistance < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(parameters->cellMaxDistance/(2.0*r))*radToDeg);
                    if( a < minAngleConstrSite )
                        minAngleConstrSite = a;
                }
            }
            qreal minAngleConstructor = 360.0;
            foreach( Cell* otherCell, constructor ) {
                qreal r = (otherCell->getRelPosition() - constructionCell->getRelPosition()).length();
                if( parameters->cellMaxDistance < (2.0*r) ) {
                    qreal a = qAbs(2.0*qAsin(parameters->cellMaxDistance/(2.0*r))*radToDeg);
                    if( a < minAngleConstructor )
                        minAngleConstructor = a;
                }
            }

            //read desired rotation angle from token
            qreal angleSum = PhysicalQuantityConverter::convertDataToAngle(tokenMem[Enums::Constr::INOUT_ANGLE]);

            //calc angular masses with respect to "constructionCell"
            qreal angMassConstrSite = 0.0;
            qreal angMassConstructor = 0.0;
            foreach( Cell* otherCell, constructionSite )
                angMassConstrSite = angMassConstrSite + (otherCell->getRelPosition() - constructionCell->getRelPosition()).lengthSquared();
            foreach( Cell* otherCell, constructor )
                angMassConstructor = angMassConstructor + (otherCell->getRelPosition() - constructionCell->getRelPosition()).lengthSquared();

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
            transformConstrSite.translate(constructionCell->getRelPosition());
            transformConstrSite.rotate(angleConstrSite, 0.0, 0.0, 1.0);
            transformConstrSite.translate(-constructionCell->getRelPosition());

            QMatrix4x4 transformConstructor;
            transformConstructor.setToIdentity();
            transformConstructor.translate(constructionCell->getRelPosition());
            transformConstructor.rotate(-angleConstructor, 0.0, 0.0, 1.0);
            transformConstructor.translate(-constructionCell->getRelPosition());

            //apply rigid transformation to construction site and constructor
            cluster->clearCellsFromMap();
			foreach(Cell* otherCell, constructionSite) {
				otherCell->setRelPosition(transformConstrSite.map(QVector3D(otherCell->getRelPosition())).toVector2D());
			}
			foreach(Cell* otherCell, constructor) {
				otherCell->setRelPosition(transformConstructor.map(QVector3D(otherCell->getRelPosition())).toVector2D());
			}

            //only rotation?
            if( performRotationOnly ) {

                //restore connection to "cell"
                cell->newConnection(constructionCell);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithoutUpdate();
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass(), cluster->getVelocity(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/parameters->cellMass_Reciprocal;

                //not enough energy?
                if( token->getEnergy() <= (parameters->cellCreationEnergy + eDiff + parameters->tokenMinEnergy + ALIEN_PRECISION) ) {
                    tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;

                    //restore cluster
                    foreach( Cell* otherCell, cluster->getCellsRef() ) {
                        otherCell->setRelPosition(relPosCells.first());
                        relPosCells.removeFirst();
                    }
                    cluster->drawCellsToMap();
                    return processingResult;
                }

                //update relative coordinates
                cluster->updateRelCoordinates(Cluster::MaintainCenter::Yes);

                //obstacle found?
                if( cmd != Enums::ConstrIn::BRUTEFORCE) {
                    bool safeMode = (cmd == Enums::ConstrIn::SAFE);
                    if( obstacleCheck(cluster, safeMode, cellMap, metric, parameters) ) {
                        tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;

                        //restore construction site
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPosition(relPosCells.first());
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
                tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS_ROT;
                tokenMem[Enums::Constr::INOUT_ANGLE] = PhysicalQuantityConverter::convertAngleToData(angleSum - angleConstrSite - angleConstructor);
                cluster->drawCellsToMap();
            }

            //not only rotation but also creating new cell
            else {

                //calc translation vector for construction site
                QVector2D transOld = constructionCell->getRelPosition() - cell->getRelPosition();
                QVector2D trans = transOld.normalized() * len;
                QVector2D transFinish;
                if( (opt == Enums::ConstrInOption::FINISH_WITH_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) )
                    transFinish = trans;


                //shift construction site
                foreach( Cell* otherCell, constructionSite )
                    otherCell->setRelPosition(otherCell->getRelPosition() + trans + transFinish);

                //calc position for new cell
                QVector2D pos = cluster->relToAbsPos(cell->getRelPosition() + transOld + transFinish);

                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVelocity(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/parameters->cellMass_Reciprocal;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::CREATE_DUP_TOKEN)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED ))
                    tokenEnergy = parameters->tokenCreationEnergy;

                //not enough energy?
                if( token->getEnergy() <= (parameters->cellCreationEnergy + tokenEnergy + eDiff + parameters->tokenMinEnergy + ALIEN_PRECISION) ) {
                    tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;

                    //restore construction site
                    foreach( Cell* otherCell, cluster->getCellsRef() ) {
                        otherCell->setRelPosition(relPosCells.first());
                        relPosCells.removeFirst();
                    }
                    cluster->drawCellsToMap();

                    //restore connection from construction site to "cell"
                    cell->newConnection(constructionCell);
                    return processingResult;
                }

                //construct new cell
                quint8 maxCon = tokenMem[Enums::Constr::IN_CELL_MAX_CONNECTIONS];
                if( maxCon < 2 )
                    maxCon = 2;
                if( maxCon > parameters->cellMaxBonds )
                    maxCon = parameters->cellMaxBonds;
                int tokenAccessNumber = tokenMem[Enums::Constr::IN_CELL_BRANCH_NO] % parameters->cellMaxTokenBranchNumber;
				quint8 metadata = tokenMem[Enums::Constr::IN_CELL_METADATA];
				Cell* newCell = constructNewCell(cell, pos, maxCon, tokenAccessNumber, metadata
					, tokenMem[Enums::Constr::IN_CELL_FUNCTION]
					, tokenMem.mid(Enums::Constr::IN_CELL_FUNCTION_DATA)
					, _context);

                //obstacle found?
                if( cmd != Enums::ConstrIn::BRUTEFORCE) {
                    bool safeMode = (cmd == Enums::ConstrIn::SAFE);
                    if( obstacleCheck(cluster, safeMode, cellMap, metric, parameters) ) {
                        tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPosition(relPosCells.first());
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
                    if( (otherCell->getNumConnections() < parameters->cellMaxBonds)
                            && (newCell->getNumConnections() < parameters->cellMaxBonds)
                            && (otherCell !=constructionCell ) ) {
                        if (metric->displacement(newCell->getRelPosition(), otherCell->getRelPosition()).length() <= (parameters->cellMaxDistance + ALIEN_PRECISION) ) {

                            //CONSTR_IN_CELL_MAX_CONNECTIONS = 0 => set "maxConnections" automatically
                            if( tokenMem.at(Enums::Constr::IN_CELL_MAX_CONNECTIONS) == 0 ) {
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
                if( (opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::CREATE_DUP_TOKEN)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
                    qreal av = averageEnergy(cell->getEnergy(), newCell->getEnergy());
                    cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //allow token access to old "construction cell"
                constructionCell->setFlagTokenBlocked(false);

                //finish construction site?
                if( (opt == Enums::ConstrInOption::FINISH_NO_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) )
                    newCell->setFlagTokenBlocked(false);

                //separate construction site?
                if( opt == Enums::ConstrInOption::FINISH_WITH_SEP ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, false);
                }
                if( (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, true);
                }

                //update token data
                tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
                tokenMem[Enums::Constr::INOUT_ANGLE] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
                if( (opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
                    if( newCell->getNumToken(true) < parameters->cellMaxToken ) {
						auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
						auto desc = TokenDescription().setEnergy(parameters->tokenCreationEnergy);
						auto newToken = factory->build(desc, _context);
                        newCell->addToken(newToken, ACTIVATE_TOKEN::Later, UPDATE_TOKEN_ACCESS_NUMBER::Yes);
                        token->setEnergy(token->getEnergy() - parameters->tokenCreationEnergy);
                    }
                }
                if( opt == Enums::ConstrInOption::CREATE_DUP_TOKEN ) {
                    if( newCell->getNumToken(true) < parameters->cellMaxToken ) {
                        auto dup = token->duplicate();
                        dup->setEnergy(parameters->tokenCreationEnergy);
                        newCell->addToken(dup, ACTIVATE_TOKEN::Later, UPDATE_TOKEN_ACCESS_NUMBER::Yes);
                        token->setEnergy(token->getEnergy() - parameters->tokenCreationEnergy);
                    }
                }
            }
        }

        //construction site connected with other cells than "cell"? => error
        else {
            tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;

            //restore connection to "cell"
            cell->newConnection(constructionCell);
        }
    }

    //start new construction site?
    else {

        //new cell connection possible?
        if( numCon < parameters->cellMaxBonds ) {

            //is there any connection?
            if( numCon > 0 ) {

                //find biggest angle gap for new cell
                QVector< qreal > angles(numCon);
                for(int i = 0; i < numCon; ++i) {
                    QVector2D displacement = cluster->calcPosition(cell->getConnection(i),true)-cluster->calcPosition(cell, true);
                    metric->correctDisplacement(displacement);
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
                angleGap = angleGap + PhysicalQuantityConverter::convertDataToAngle(tokenMem[Enums::Constr::INOUT_ANGLE]);

                //calc coordinates for new cell from angle gap and construct cell
                QVector2D angleGapPos = Physics::unitVectorOfAngle(angleGap)*parameters->cellFunctionConstructorOffspringDistance;
                QVector2D pos = cluster->calcPosition(cell)+angleGapPos;
                if( (opt == Enums::ConstrInOption::FINISH_WITH_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) )
                    pos = pos + angleGapPos;


                //estimate expended energy for new cell
                qreal kinEnergyOld = Physics::kineticEnergy(cluster->getMass(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel());
                qreal angularMassNew = cluster->calcAngularMassWithNewParticle(pos);
                qreal angularVelNew = Physics::newAngularVelocity(cluster->getAngularMass(), angularMassNew, cluster->getAngularVel());
                qreal kinEnergyNew = Physics::kineticEnergy(cluster->getMass()+1.0, cluster->getVelocity(), angularMassNew, angularVelNew);
                qreal eDiff = (kinEnergyNew-kinEnergyOld)/parameters->cellMass_Reciprocal;

                //energy for possible new token
                qreal tokenEnergy = 0;
                if( (opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::CREATE_DUP_TOKEN )
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) )
                    tokenEnergy = parameters->tokenCreationEnergy;

                //not enough energy?
                if( token->getEnergy() <= (parameters->cellCreationEnergy + tokenEnergy + eDiff + parameters->tokenMinEnergy + ALIEN_PRECISION) ) {
                    tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
                    return processingResult;
                }

                //construct new cell
                cluster->clearCellsFromMap();
                quint8 maxCon = tokenMem[Enums::Constr::IN_CELL_MAX_CONNECTIONS];
                if( maxCon < 1 )
                    maxCon = 1;
                if( maxCon > parameters->cellMaxBonds )
                    maxCon = parameters->cellMaxBonds;
                int tokenAccessNumber = tokenMem[Enums::Constr::IN_CELL_BRANCH_NO]
                        % parameters->cellMaxTokenBranchNumber;
				quint8 metadata = tokenMem[Enums::Constr::IN_CELL_METADATA];
				Cell* newCell = constructNewCell(cell, pos, maxCon, tokenAccessNumber, metadata
					, tokenMem[Enums::Constr::IN_CELL_FUNCTION]
					, tokenMem.mid(Enums::Constr::IN_CELL_FUNCTION_DATA), _context);

                //obstacle found?
                if( cmd != Enums::ConstrIn::BRUTEFORCE) {
                    bool safeMode = (cmd == Enums::ConstrIn::SAFE);
                    if( obstacleCheck(cluster, safeMode, cellMap, metric, parameters) ) {
                        tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_OBSTACLE;

                        //restore construction site
                        cluster->removeCell(newCell);
                        delete newCell;
                        foreach( Cell* otherCell, cluster->getCellsRef() ) {
                            otherCell->setRelPosition(relPosCells.first());
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
                if( (opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::CREATE_DUP_TOKEN)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
                    qreal av = averageEnergy(cell->getEnergy(), newCell->getEnergy());
                    cell->setEnergy(av);
                    newCell->setEnergy(av);
                }

                //update angular velocity
                cluster->setAngularVel(angularVelNew);

                //finish construction site?
                if( (opt == Enums::ConstrInOption::FINISH_NO_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) )
                    newCell->setFlagTokenBlocked(false);

                //separate construction site?
                if( opt == Enums::ConstrInOption::FINISH_WITH_SEP ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, false);
                }
                if( (opt == Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
                    processingResult.decompose = true;
                    separateConstruction(newCell, cell, true);
                }

                //update token data
                tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
                tokenMem[Enums::Constr::INOUT_ANGLE] = 0;
                cluster->drawCellsToMap();

                //create new token if desired
				auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
				if ((opt == Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                        || (opt == Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED) ) {
					auto desc = TokenDescription().setEnergy(parameters->tokenCreationEnergy);
					auto token = factory->build(desc, _context);
                    newCell->addToken(token, ACTIVATE_TOKEN::Later, UPDATE_TOKEN_ACCESS_NUMBER::Yes);
                    token->setEnergy(token->getEnergy() - parameters->tokenCreationEnergy);
                }
                if( opt == Enums::ConstrInOption::CREATE_DUP_TOKEN ) {
                    auto dup = token->duplicate();
                    dup->setEnergy(parameters->tokenCreationEnergy);
                    newCell->addToken(dup, ACTIVATE_TOKEN::Later, UPDATE_TOKEN_ACCESS_NUMBER::Yes);
                    token->setEnergy(token->getEnergy() - parameters->tokenCreationEnergy);
                }

            }

            //no connection by now
            else {
                tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
            }
        }

        //no new connection possible
        else {

            //error code
            tokenMem[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_CONNECTION;
        }
    }
    return processingResult;
}

