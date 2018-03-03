#include <QMatrix4x4>
#include <qmath.h>

#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"
#include "Model/Local/Cell.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Local/CellFeatureChain.h"
#include "Model/Local/EntityFactory.h"
#include "Model/Local/Token.h"
#include "Model/Local/Particle.h"
#include "Model/Local/Physics.h"
#include "Model/Api/Settings.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/SpacePropertiesLocal.h"
#include "Model/Api/SimulationParameters.h"

#include "Cluster.h"

const int PROTECTION_COUNTER_AFTER_COLLISION = 14;

Cluster::Cluster(QList< Cell* > cells, uint64_t id, qreal angle, QVector2D pos, qreal angularVel
    , QVector2D vel, UnitContext* context)
    : EntityWithTimestamp(context), _id(id), _angle(angle), _angularVel(angularVel), _vel(vel), _cells(cells)
{
	setCenterPosition(pos);
    foreach(Cell* cell, _cells) {
        cell->setCluster(this);
    }
    updateTransformationMatrix();
    updateRelCoordinates();
    updateAngularMass();
}

namespace
{
	QVector2D calcCenterPosition(QList<Cell*> const& cells)
	{
		QVector2D result;
		foreach(Cell* cell, cells) {
			result += cell->getCluster()->calcPosition(cell);
		}
		result /= cells.size();
		return result;
	}

	void setRelPositionInCluster(QList<Cell*> const& cells, Cluster* cluster)
	{
		foreach(Cell* cell, cells) {

			//adjust relative position of the cells
			QVector2D pos(cell->getCluster()->calcPosition(cell));
			cell->setCluster(cluster);
			cell->setAbsPosition(pos);
		}
	}
}

Cluster::Cluster(QList< Cell* > cells, qreal angle, UnitContext* context)
    : EntityWithTimestamp(context), _angle(angle), _cells(cells)
{
	_id = _context->getNumberGenerator()->getTag();
	setCenterPosition(calcCenterPosition(_cells));
	setRelPositionInCluster(_cells, this);
    updateAngularMass();
    updateVel_angularVel_via_cellVelocities();
}

Cluster::~Cluster ()
{
    foreach(Cell* cell, _cells) {
        delete cell;
    }
}

void Cluster::clearCellsFromMap ()
{
	auto cellMap = _context->getCellMap();
    foreach( Cell* cell, _cells) {
		cellMap->removeCellIfPresent(applyTransformation(cell->getRelPosition()), cell);
    }
}

void Cluster::setContext(UnitContext * context)
{
	_context = context;
	for(auto const& cell : _cells) {
		cell->setContext(context);
	}
}

ClusterDescription Cluster::getDescription(ResolveDescription const& resolveDescription) const
{
	ClusterDescription result;
	result.setPos(_pos).setVel(_vel).setAngle(_angle).setAngularVel(_angularVel).setMetadata(_meta);
	if (resolveDescription.resolveIds) {
		result.setId(_id);
	}
	for (auto const& cell : _cells) {
		result.addCell(cell->getDescription(resolveDescription));
	}
	if (!result.cells) {
		result.cells = vector<CellDescription>();
	}
	return result;
}

void Cluster::applyChangeDescription(ClusterChangeDescription const & change)
{
	if (change.pos) {
		setCenterPosition(*change.pos);
	}
	if (change.vel) {
		setVelocity(*change.vel);
	}
	if (change.angle) {
		setAngle(*change.angle);
	}
	if (change.angularVel) {
		setAngularVel(*change.angularVel);
	}
	if (change.metadata) {
		setMetadata(*change.metadata);
	}
	if (!change.cells.empty()) {
		map<uint64_t, Cell*> cellsByIds;
		for (auto const& cell : _cells) {
			cellsByIds.insert_or_assign(cell->getId(), cell);
		}
		for (auto const& cellT : change.cells) {
			if (cellT.isModified()) {
				if (cellsByIds.find(cellT->id) != cellsByIds.end()) {
					cellsByIds.at(cellT->id)->applyChangeDescription(cellT.getValue());
				}
			}
		}
	}
}

void Cluster::clearCellFromMap (Cell* cell)
{
	auto cellMap = _context->getCellMap();
	cellMap->removeCellIfPresent(applyTransformation(cell->getRelPosition()), cell);
}

void Cluster::drawCellsToMap ()
{
	auto cellMap = _context->getCellMap();
	foreach(Cell* cell, _cells) {
        QVector2D pos(calcPosition(cell, true));
		cellMap->setCell(pos, cell);
    }
}

//initiate movement of particles
void Cluster::processingInit ()
{
	if (!isTimestampFitting()) {
		return;
	}

    //clear cells
	auto cellMap = _context->getCellMap();
	foreach(Cell* cell, _cells) {

        //remove particle from old position
        //-> note that due to numerical effect during fusion position can be slightly changed
		cellMap->removeCellIfPresent(applyTransformation(cell->getRelPosition()), cell);
        if( cell->getProtectionCounter() > 0 )
            cell->setProtectionCounter(cell->getProtectionCounter()-1);
    }
}

//dissipation, returns lost energy
void Cluster::processingDissipation (QList< Cluster* >& fragments, QList< Particle* >& energyParticles)
{
	if (!isTimestampFitting()) {
		return;
	}
	
	updateCellVel();
	auto parameters = _context->getSimulationParameters();

    //determine energies (the new kinetic energy will be calculated later)
    qreal oldEnergy = Physics::kineticEnergy(_cells.size(), _vel, _angularMass, _angularVel);
    qreal newEnergy = 0.0;

    //dissipation
    bool cellDestroyed(false);
    QMutableListIterator<Cell*> i(_cells);
    while (i.hasNext()) {
        Cell* cell(i.next());
        Particle* energyParticle(0);

        //radiation of cell
        qreal cellEnergy = cell->getEnergy();
        radiation(cellEnergy, cell, energyParticle);
        cell->setEnergy(cellEnergy);
        if( energyParticle )
            energyParticles << energyParticle;

        //radiation of tokens
        int numToken = cell->getNumToken();
        if( numToken > 0 ) {
            for( int i = 0; i < numToken; ++i ) {
				qreal tokenEnergy = cell->getToken(i)->getEnergy();
                radiation(tokenEnergy, cell, energyParticle);
				cell->getToken(i)->setEnergy(tokenEnergy);
                if( energyParticle )
                    energyParticles << energyParticle;
            }
        }

        //kill cell?
        if( (cell->isToBeKilled() || (cell->getEnergy() < parameters->cellMinEnergy)) ) {
            qreal kinEnergy = Physics::kineticEnergy(1.0, cell->getVelocity(), 0.0, 0.0);
            qreal internalEnergy = cell->getEnergyIncludingTokens();
            EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
            qreal energyForParticle = internalEnergy + kinEnergy / parameters->cellMass_Reciprocal;

			QVector2D pos = calcPosition(cell, true);
			QVector2D vel = cell->getVelocity();
			auto desc = ParticleDescription().setEnergy(energyForParticle).setPos(QVector2D(pos.x(), pos.y())).setVel(QVector2D(vel.x(), vel.y()));
            energyParticle = factory->build(desc, _context);
			ParticleMetadata metadata;
			metadata.color = cell->getMetadata().color;
            energyParticle->setMetadata(metadata);
            energyParticles << energyParticle;
            cell->setEnergy(0.0);

            //remove cell and all references
            cell->delAllConnection();
            clearCellFromMap(cell);
            delete cell;
            cellDestroyed = true;
            i.remove();

            newEnergy += kinEnergy;
        }
    }

    if( _cells.isEmpty() )
        return;

    //decomposition of the cluster
    if( cellDestroyed ) {

        int size = _cells.size();
        do {

            //find fragment
            QList< Cell* > component;
			quint64 tag = _context->getNumberGenerator()->getTag();
            getConnectedComponent(_cells[0], tag, component);
            if( component.size() < size ) {
                EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
                Cluster* part = new Cluster(component, _angle, _context);
                fragments << part;

                //remove fragment from cluster
                QMutableListIterator<Cell*> i(_cells);
                while (i.hasNext()) {
                    if( i.next()->getTag() == tag )
                        i.remove();
                }
            }
            else {
                updateRelCoordinates();
                updateAngularMass();
                updateVel_angularVel_via_cellVelocities();

                //calc energy difference
                newEnergy += Physics::kineticEnergy(size, _vel, _angularMass, _angularVel);
                qreal diffEnergy = oldEnergy-newEnergy;

                //spread energy difference on cells
                qreal diffEnergyCell = (diffEnergy/static_cast<qreal>(size)) / parameters->cellMass_Reciprocal;
                foreach(Cell* cell, _cells) {
                    if( cell->getEnergy() > (-diffEnergyCell) )
                        cell->setEnergy(cell->getEnergy() + diffEnergyCell);
                }
                return;
            }
        }
        while( !_cells.isEmpty() );

        //calc energy difference
        foreach(Cluster* cluster, fragments) {
            newEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel());
        }
        qreal diffEnergy = oldEnergy-newEnergy;

        //spread energy difference on cells
        qreal diffEnergyCell = (diffEnergy/static_cast<qreal>(size)) / parameters->cellMass_Reciprocal;
        foreach(Cluster* cluster, fragments)
            foreach(Cell* cell, cluster->getCellsRef()) {
                if( cell->getEnergy() > (-diffEnergyCell) )
                    cell->setEnergy(cell->getEnergy() + diffEnergyCell);
            }
    }
}

void Cluster::processingMutationByChance()
{
	if (!isTimestampFitting()) {
		return;
	}
	
	foreach(Cell* cell, _cells) {
		cell->mutationByChance();
	}
}

void Cluster::processingMovement ()
{
	if (!isTimestampFitting()) {
		return;
	}
	
	struct CollisionData {
        int movementState = 0;  //0: will do nothing, 1: collision, 2: fusion
        CellSet overlappingCells;
        QList< QPair< Cell*, Cell* > > overlappingCellPairs;
    };
	auto parameters = _context->getSimulationParameters();
	auto metric = _context->getSpaceProperties();
	auto cellMap = _context->getCellMap();

	_angle += _angularVel;
    if( _angle > 360.0 )
        _angle -= 360.0;
    if( _angle < 0.0 )
        _angle += 360.0;
    _pos += _vel;
    metric->correctPosition(_pos);
    updateTransformationMatrix();
    QVector2D pos;

    //collect information for every colliding cluster
    QMap< quint64, CollisionData > clusterCollisionDataMap;
    QMap< quint64, Cell* > idCellMap;
    QMap< quint64, Cluster* > idClusterMap;
	foreach(Cell* cell, _cells) {
        pos = calcPosition(cell, true);
        for(int x = -1; x < 2; ++x)
            for(int y = -1; y < 2; ++y) {
                Cell* tempCell = cellMap->getCell(pos+QVector2D(x, y));
                if( tempCell )
                    if( tempCell->getCluster() != this ) {

                        //cell close enough?
                        QVector2D displacement(tempCell->getCluster()->calcPosition(tempCell, true)-pos);
                        metric->correctDisplacement(displacement);
                        if( displacement.length() < parameters->cellMaxDistance ) {
                            quint64 clusterId = tempCell->getCluster()->getId();

                            //read collision data for the colliding cluster
                            CollisionData colData;
                            if( clusterCollisionDataMap.contains(clusterId) )
                                colData = clusterCollisionDataMap[clusterId];
                            else
                                colData.movementState = 0;

                            //remember cell
                            idCellMap[tempCell->getId()] = tempCell;
                            idClusterMap[clusterId] = tempCell->getCluster();
                            colData.overlappingCells.insert(tempCell);
                            colData.overlappingCellPairs << QPair< Cell*, Cell* >(cell, tempCell);

                            //first time check?
                            if( colData.movementState == 0 ) {

                                //fusion possible? (velocities high enough?)
								if (cell->connectable(tempCell) && ((cell->getVelocity() - tempCell->getVelocity()).length() >= parameters->cellFusionVelocity)
									&& connectable(tempCell->getCluster())) {
									colData.movementState = 2;
								}

                                //collision possible?
                                else if( cell->getProtectionCounter() == 0 && tempCell->getProtectionCounter() == 0 )
                                    colData.movementState = 1;

                            }

                            //cluster already set for collision?
                            if( colData.movementState == 1 ) {

                                //fusion possible?
								if (cell->connectable(tempCell) && ((cell->getVelocity() - tempCell->getVelocity()).length() >= parameters->cellFusionVelocity)
									&& connectable(tempCell->getCluster())) {
									colData.movementState = 2;
								}
                            }

                            //update collision data
                            clusterCollisionDataMap[clusterId] = colData;
                        }
                    }
            }
    }

    //process information of the colliding clusters
    QMapIterator< quint64, CollisionData > it(clusterCollisionDataMap);
    while(it.hasNext()) {
        it.next();
        Cluster* otherCluster = idClusterMap[it.key()];
        CollisionData collisionData = it.value();

        //collision?
        if( collisionData.movementState == 1 ) {

            //set protection counter to avoid collision in the next few time steps
            QListIterator< QPair< Cell*, Cell* > > it2(collisionData.overlappingCellPairs);
            while( it2.hasNext() ) {
                QPair< Cell*, Cell* > cellPair(it2.next());
                cellPair.first->setProtectionCounter(PROTECTION_COUNTER_AFTER_COLLISION);
                cellPair.second->setProtectionCounter(PROTECTION_COUNTER_AFTER_COLLISION);
            }

            //performing collisions:
            //calc center position of the overlapping cells
            QVector2D centerPos;
            for (Cell* otherCell : collisionData.overlappingCells) {
                centerPos = centerPos + otherCluster->calcPosition(otherCell);
            }
            centerPos = centerPos/collisionData.overlappingCells.size();

            //calc negative velocity at the center position (later used as outerSpace vector)
            QVector2D rAPp = centerPos-_pos;
            metric->correctDisplacement(rAPp);
            rAPp = Physics::rotateQuarterCounterClockwise(rAPp);
            QVector2D rBPp = centerPos-otherCluster->getPosition();
            metric->correctDisplacement(rBPp);
            rBPp = Physics::rotateQuarterCounterClockwise(rBPp);
            QVector2D outwardVector = (otherCluster->getVelocity()-rBPp*otherCluster->getAngularVel()*degToRad)-(_vel-rAPp*_angularVel*degToRad);

            //calc center normal vector of the overlapping cells from the other cluster
            QVector2D n;
            for (Cell* otherCell : collisionData.overlappingCells) {
                n = n + otherCell->calcNormal(outwardVector).normalized();
            }

            //calc new vectors
            qreal mA = _cells.size();
            qreal mB = otherCluster->getCellsRef().size();
            QVector2D vA2, vB2;
            qreal angularVelA2 = 0;
            qreal angularVelB2 = 0;
            n.normalize();
            if( n.length() < ALIEN_PRECISION )
                n.setX(1.0);

            Physics::collision(_vel, otherCluster->getVelocity(),//, clusterPos, otherClusterPos, centerPos,
                               rAPp, rBPp,
                               _angularVel, otherCluster->getAngularVel(), n,
                               _angularMass, otherCluster->getAngularMass(), mA, mB, vA2, vB2, angularVelA2,
                               angularVelB2);

            //set new vectors
            _vel = vA2;
            otherCluster->setVelocity(vB2);
            _angularVel = angularVelA2;
            otherCluster->setAngularVel(angularVelB2);

        }

        //fusion?
        if( collisionData.movementState == 2 ) {

            //connecting clusters if possible
            CellSet fusedCells;
            QListIterator< QPair< Cell*, Cell* > > it2(collisionData.overlappingCellPairs);
            while (it2.hasNext()) {
                QPair< Cell*, Cell* > item(it2.next());
                Cell* cell(item.first);
                Cell* otherCell(item.second);
                QVector2D displacement(otherCell->getCluster()->calcPosition(otherCell, true)-calcPosition(cell, true));
                metric->correctDisplacement(displacement);

                //kill cell if too close
                if( displacement.length() < parameters->cellMinDistance ){
                    if( _cells.size() > otherCell->getCluster()->getCellsRef().size()) {
    //                    if( otherCell->_protectionCounter == 0 ) {
                            otherCell->setToBeKilled(true);
    //                    }
                    }
                    else {
    //                    if( cell->_protectionCounter == 0 ) {
                        otherCell->setToBeKilled(true);
    //                    }
                    }
                }

                //connecting cells
                if( cell->connectable(otherCell) ) {
                    cell->newConnection(otherCell);
                    fusedCells.insert(cell);
                    fusedCells.insert(otherCell);
                    idCellMap[cell->getId()] = cell;
                }
                otherCluster = otherCell->getCluster();
            }

            //cells connected?
            if (!fusedCells.empty()) {
                updateCellVel();
                otherCluster->updateCellVel();

                //calc old kinetic energy of both clusters
                qreal mA = _cells.size();
                qreal mB = otherCluster->getCellsRef().size();
                qreal eKinOld1 = Physics::kineticEnergy(mA, _vel, _angularMass, _angularVel);
                qreal eKinOld2 = Physics::kineticEnergy(mB, otherCluster->getVelocity(), otherCluster->getAngularMass(), otherCluster->getAngularVel());

                //calculate new center
                QVector2D center;
                QVector2D correction(metric->correctionIncrement(_pos, otherCluster->getPosition()));
                foreach( Cell* cell, _cells) {
                    cell->setRelPosition(calcPosition(cell));     //store absolute position only temporarily
                    center += cell->getRelPosition();
                }
                foreach( Cell* cell, otherCluster->getCellsRef()) {
                    cell->setRelPosition(otherCluster->calcPosition(cell)+correction);
                    center += cell->getRelPosition();
                }
                center /= (_cells.size()+otherCluster->getCellsRef().size());
                _pos = center;
                updateTransformationMatrix();

                //transfer cells
                QList< Cell* > cells(otherCluster->getCellsRef());
                _cells << cells;
                otherCluster->getCellsRef().clear();
                foreach( Cell* cell, cells) {
                    cell->setCluster(this);
                }

                //set relative coordinates
                foreach( Cell* cell, _cells) {
                    cell->setRelPosition(absToRelPos(cell->getRelPosition()));
                }
                metric->correctPosition(_pos);
                updateTransformationMatrix();

                //calc angular mass, velocity, angular velocity
                updateAngularMass();
                updateVel_angularVel_via_cellVelocities();

                //calc newkinetic energy of united cluster
                qreal eKinNew = Physics::kineticEnergy(_cells.size(), _vel, _angularMass, _angularVel);

                //spread lost kinetic energy to tokens and internal energy of the fused cells
                qreal eDiff = ((eKinOld1 + eKinOld2 - eKinNew) / static_cast<qreal>(fusedCells.size())) / parameters->cellMass_Reciprocal;
                if( eDiff > ALIEN_PRECISION ) {
                    for (Cell* cell : fusedCells) {

                        //create token?
                        if( (cell->getNumToken() < parameters->cellMaxToken) && (eDiff > parameters->tokenMinEnergy) ) {
							auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
							int tokenMemSize = _context->getSimulationParameters()->tokenMemorySize;
							auto desc = TokenDescription().setEnergy(eDiff).setData(_context->getNumberGenerator()->getRandomArray(tokenMemSize));
                            auto token = factory->build(desc, _context);
                            cell->addToken(token, Cell::ActivateToken::Now, Cell::UpdateTokenBranchNumber::Yes);
                        }
                        //if not add to internal cell energy
                        else
                            cell->setEnergy(cell->getEnergy() + eDiff);
                    }
                }
            }
        }
    }

    //draw new cells
    foreach( Cell* cell, _cells) {
        QVector2D pos = applyTransformation(cell->getRelPosition());
        cellMap->setCell(pos, cell);
    }

}

//token processing
void Cluster::processingToken (QList< Particle* >& energyParticles, bool& decompose)
{
	if (!isTimestampFitting()) {
		return;
	}

	auto parameters = _context->getSimulationParameters();
	vector<Token*> spreadToken(parameters->cellMaxBonds);
    vector<Cell*> spreadTokenCells(parameters->cellMaxBonds);

    //placing new tokens
    foreach(Cell* cell, _cells) {
        while(true) {
            Token* token = cell->takeTokenFromStack();
            if( !token )
                break;
            int tokenAccessNumber = token->getTokenAccessNumber();

            //determine number of places for tokens
            int numPlaces = 0;
            for(int j = 0; j < cell->getNumConnections(); ++j) {
                Cell* otherCell = cell->getConnection(j);
                if( (((tokenAccessNumber+1) % parameters->cellMaxTokenBranchNumber) == otherCell->getBranchNumber()) && (!otherCell->isTokenBlocked())
                    && (otherCell->getNumToken(true) < parameters->cellMaxToken ) ) {
                    ++numPlaces;
                }
            }

            //no free places for token?
            if( numPlaces == 0 ) {
                cell->setEnergy(cell->getEnergy() + token->getEnergy());
                delete token;
            }

            //free places for tokens available
            else {
                //not enough cell energy available?
                if( //(cell->_energy < ((qreal)numPlaces-1.0)*token->energy) ||
                    token->getEnergy() < parameters->tokenMinEnergy) {
                    cell->setEnergy(cell->getEnergy() + token->getEnergy());
                    delete token;
                }
                else {
                    //calc available token energy
                    qreal tokenEnergy = token->getEnergy();
                    qreal availableTokenEnergy = tokenEnergy / numPlaces;

                    //spread token to free places on adjacent cells and duplicate token if necessary
                    int spreadTokenCounter = 0;
                    for(int j = 0; j < cell->getNumConnections(); ++j) {
                        Cell* otherCell = cell->getConnection(j);
                        if( (((tokenAccessNumber+1) % parameters->cellMaxTokenBranchNumber) == otherCell->getBranchNumber()) && (!otherCell->isTokenBlocked())
                            && (otherCell->getNumToken(true) < parameters->cellMaxToken ) ) {
                            if( spreadTokenCounter > 0 ) {
                                spreadTokenCells[spreadTokenCounter] = otherCell;
                                spreadToken[spreadTokenCounter] = token->duplicate();
                                otherCell->addToken(spreadToken[spreadTokenCounter], Cell::ActivateToken::Later, Cell::UpdateTokenBranchNumber::Yes);
                            }
                            if( spreadTokenCounter == 0 ) {
                                spreadTokenCells[0] = otherCell;
                                spreadToken[0] = token;
                                otherCell->addToken(token, Cell::ActivateToken::Later, Cell::UpdateTokenBranchNumber::Yes);
                            }
                            if( numPlaces > 1 ) {
                                spreadToken[spreadTokenCounter]->setEnergy(availableTokenEnergy);

                                //transfer remaining energy from cell to token if possible
                                if (otherCell->getEnergy() > parameters->cellMinEnergy + tokenEnergy - availableTokenEnergy) {
                                    spreadToken[spreadTokenCounter]->setEnergy(tokenEnergy);
                                    otherCell->setEnergy(otherCell->getEnergy() - (tokenEnergy-availableTokenEnergy));
                                }
                                else if (otherCell->getEnergy() > parameters->cellMinEnergy) {
									spreadToken[spreadTokenCounter]->setEnergy(spreadToken[spreadTokenCounter]->getEnergy()
										+ otherCell->getEnergy() - parameters->cellMinEnergy);
                                    otherCell->setEnergy(parameters->cellMinEnergy);
                                }
                            }
                            spreadTokenCounter++;
                        }
                    }


                    //execute cell functions and token energy guidance system on the cells with the tokens
                    for( int i = 0; i < spreadTokenCounter; ++i ) {

                        //execute cell function
                        CellFeatureChain::ProcessingResult processingResult = spreadTokenCells[i]->getFeatures()->process(spreadToken[i], spreadTokenCells[i], cell);
                        if( processingResult.decompose )
                            decompose = true;
                        if( processingResult.newEnergyParticle )
                            energyParticles << processingResult.newEnergyParticle;

                    }

                    //average internal energies
                    qreal energyAv = cell->getEnergy();
                    for( int i = 0; i < spreadTokenCounter; ++i )
                        energyAv += spreadTokenCells[i]->getEnergy();
                    energyAv = energyAv / (spreadTokenCounter+1);
                    for( int i = 0; i < spreadTokenCounter; ++i )
                        spreadTokenCells[i]->setEnergy(energyAv);
                    cell->setEnergy(energyAv);
                }
            }
        }
    }
}

//activate new token and kill cells which are too close or where too much forces are applied
void Cluster::processingCompletion ()
{
	if (!isTimestampFitting()) {
		return;
	}

	auto metric = _context->getSpaceProperties();
	auto cellMap = _context->getCellMap();
	qreal maxClusterRadius = qMin(metric->getSize().x / 2.0, metric->getSize().y / 2.0);
    foreach( Cell* cell, _cells) {

        //activate tokens
        cell->activatingNewTokens();

        //kill cells which are too far from cluster center
        if(cell->getRelPosition().length() > (maxClusterRadius-1.0) )
            cell->setToBeKilled(true);

        //find nearby cells and kill if they are too close
        QVector2D pos = calcPosition(cell, true);
        for( int x = -1; x < 2; ++x )
            for( int y = -1; y < 2; ++y ) {
                Cell* otherCell(cellMap->getCell(pos+QVector2D(x, y)));
                if( otherCell ) {
                    if( otherCell != cell ) {
//                    if( otherCell->_cluster != this ) {
                        Cluster* otherCluster = otherCell->getCluster();
//                        foreach(Cell* otherCell2, otherCluster->getCellsRef()) {
//                            if( otherCell2 != cell ) {
                                QVector2D displacement = otherCluster->calcPosition(otherCell, true)-calcPosition(cell, true);
                                metric->correctDisplacement(displacement);
                                if (displacement.length() < _context->getSimulationParameters()->cellMinDistance) {
                                    if( _cells.size() > otherCluster->getCellsRef().size()) {
//                                        if( otherCell->_protectionCounter == 0 ) {
                                            otherCell->setToBeKilled(true);
//                                        }
                                    }
                                    else {
//                                        if( cell->_protectionCounter == 0 ) {
                                            cell->setToBeKilled(true);
//                                        }
                                    }
                                }
                        //    }
                        //}
                    }
                }
            }
    }
}

void Cluster::addCell (Cell* cell, QVector2D absPos, UpdateInternals update /*= UpdateInternals::Yes*/)
{
    cell->setRelPosition(absToRelPos(absPos));
    cell->setCluster(this);
    _cells << cell;

	if (update == Cluster::UpdateInternals::Yes) {
		updateInternals(MaintainCenter::Yes);
	}
}

void Cluster::removeCell (Cell* cell, MaintainCenter maintainCenter /*= MaintainCenter::Yes*/)
{
    cell->delAllConnection();
    _cells.removeAll(cell);

	updateInternals(maintainCenter);
}

void Cluster::updateCellVel (bool forceCheck)
{
    if( _cells.size() == 1 ) {
        _cells[0]->setVelocity(_vel);
    }
    else {

        //calc cell velocities
		auto parameters = _context->getSimulationParameters();
		foreach(Cell* cell, _cells) {
            QVector2D vel = Physics::tangentialVelocity(calcCellDistWithoutTorusCorrection(cell), _vel, _angularVel);
            if( cell->getVelocity().isNull() ) {
                cell->setVelocity(vel);
            }
            else {
                QVector2D a = vel - cell->getVelocity();

                //destroy cell if acceleration exceeds a certain threshold
                if( forceCheck ) {
                    if (a.length() > parameters->callMaxForce) {
                        if (_context->getNumberGenerator()->getRandomReal() < parameters->cellMaxForceDecayProb)
                            cell->setToBeKilled(true);
                    }
                }
                cell->setVelocity(vel);
            }
        }
    }
}

void Cluster::updateAngularMass () {

    //calc angular mass
    _angularMass = 0.0;
    foreach( Cell* cell, _cells)
        _angularMass += (cell->getRelPosition().lengthSquared());
}

void Cluster::updateRelCoordinates (MaintainCenter maintainCenter /*= MaintainCenter::No*/)
{
	if (_cells.isEmpty()) {
		return;
	}
    if( maintainCenter == MaintainCenter::Yes) {

        //calc new center in relative coordinates
//        calcTransform();
        QVector2D center;
        foreach( Cell* cell, _cells) {
            center += cell->getRelPosition();
        }
        center /= _cells.size();

        //set rel coordinated with respect to the new center
        foreach( Cell* cell, _cells) {
            cell->setRelPosition(cell->getRelPosition() - center);
        }
    }
    else {

        //center transformation
		QMatrix4x4 oldTransform(_transform);
		setCenterPosition(calcCenterPosition(_cells));
        QMatrix4x4 newTransformInv(_transform.inverted());

        //set rel coordinated with respect to the new center
        foreach( Cell* cell, _cells) {
            cell->setRelPosition(applyTransformation(newTransformInv, applyTransformation(oldTransform, cell->getRelPosition())));
        }
    }
}

//Note: angular mass needs to be calculated before, energy may be lost
void Cluster::updateVel_angularVel_via_cellVelocities ()
{
    if( _cells.size() > 1 ) {

        //first step: calc cluster mean velocity
        _vel = QVector2D();
        foreach( Cell* cell, _cells ) {
            _vel += cell->getVelocity();
        }
        _vel = _vel/_cells.size();

        //second step: calc angular momentum for the cluster in the inertia system with velocity _vel
        qreal angularMomentum = 0.0;
        foreach( Cell* cell, _cells ) {
            QVector2D r = calcPosition(cell)-_pos;
            QVector2D v = cell->getVelocity() - _vel;
            angularMomentum += Physics::angularMomentum(r, v);     //we only need the 3rd component of the 3D cross product
        }

        //third step: calc angular velocity via the third component of the angular momentum
        _angularVel = Physics::angularVelocity(angularMomentum, _angularMass);

    }
    else if( _cells.size() == 1 ) {
        _vel = _cells[0]->getVelocity();
        _angularVel = 0.0;
    }
}


QVector2D Cluster::calcPosition (const Cell* cell, bool metricCorrection) const
{
    QVector2D cellPos = applyTransformation(cell->getRelPosition());
	if (metricCorrection) {
		_context->getSpaceProperties()->correctPosition(cellPos);
	}
    return cellPos;
}

QVector2D Cluster::calcCellDistWithoutTorusCorrection (Cell* cell) const
{
    return calcPosition(cell)-_pos;
}

QList< Cluster* > Cluster::decompose () const
{
	auto numberGen = _context->getNumberGenerator();
	QList< Cluster* > fragments;
    while( !_cells.isEmpty() ) {


        //find fragment
        QList< Cell* > component;
        quint64 tag(numberGen->getTag());
        getConnectedComponent(_cells[0], tag, component);

        //remove fragment from clusters
        QMap< quint64, Cluster* > idClusterMap;
        foreach( Cell* cell, component) {
            idClusterMap[cell->getCluster()->getId()] = cell->getCluster();
        }

        foreach( Cluster* cluster, idClusterMap.values() ) {
            QMutableListIterator<Cell*> i(cluster->getCellsRef());
            while (i.hasNext()) {
                if( i.next()->getTag() == tag )
                    i.remove();
            }
        }
        fragments << new Cluster(component, _angle, _context);
    }
    return fragments;
}

qreal Cluster::calcAngularMassWithNewParticle (QVector2D particlePos) const
{

    //calc new center
    QVector2D particleRelPos = absToRelPos(particlePos);
    QVector2D center = particleRelPos;
    foreach(Cell* cell, _cells) {
        center = center + cell->getRelPosition();
    }
    center = center / (_cells.size()+1);

    //calc new angular mass
	SpacePropertiesLocal* metric = _context->getSpaceProperties();
    QVector2D diff = particleRelPos - center;
	metric->correctDisplacement(diff);
    qreal aMass = diff.lengthSquared();
    foreach(Cell* cell, _cells) {
        diff = cell->getRelPosition() - center;
		metric->correctDisplacement(diff);
        aMass = aMass + diff.lengthSquared();
    }
    return aMass;
}

qreal Cluster::calcAngularMassWithoutUpdate () const
{

    //calc new center
    QVector2D center;
    foreach(Cell* cell, _cells) {
        center = center + cell->getRelPosition();
    }
    center = center / (_cells.size());

    //calc new angular mass
    qreal aMass = 0.0;
	SpacePropertiesLocal* metric = _context->getSpaceProperties();
	foreach(Cell* cell, _cells) {
        QVector2D displacement = cell->getRelPosition() - center;
		metric->correctDisplacement(displacement);
        aMass = aMass + displacement.lengthSquared();
    }
    return aMass;
}

double Cluster::calcLinearKineticEnergy() const
{
	double mass = getMass();
	QVector2D vel = getVelocity();
	return 0.5 * mass * vel.lengthSquared();
}

double Cluster::calcRotationalKineticEnergy() const
{
	return 0.0;
}

bool Cluster::isEmpty() const
{
    return _cells.isEmpty();
}

QList< Cell* >& Cluster::getCellsRef ()
{
    return _cells;
}

/*QVector2D CellClusterImpl::getCoordinate (Cell* cell)
{
    return _transform.map(cell->getRelPosition());
}
*/

void Cluster::findNearestCells (QVector2D pos, Cell*& cell1, Cell*& cell2) const
{
    qreal bestR1(0.0);
    qreal bestR2(0.0);
    cell1 = 0;
    cell2 = 0;
    foreach( Cell* cell, _cells) {
        qreal r = (calcPosition(cell, true)-pos).lengthSquared();
        if(r < 1.0) {
            if( (r < bestR1) || (!cell1) ) {
                cell2 = cell1;
                bestR2 = bestR1;
                cell1 = cell;
                bestR1 = r;
            }
            else if( (r < bestR2) || (!cell2) ) {
                cell2 = cell;
                bestR2 = r;
            }
        }
    }
}

const quint64& Cluster::getId () const
{
    return _id;
}

void Cluster::setId (quint64 id)
{
    _id = id;
}

QList< quint64 > Cluster::getCellIds () const
{
    QList< quint64 > ids;
    foreach( Cell* cell, _cells )
        ids << cell->getId();
    return ids;
}

QVector2D Cluster::getPosition () const
{
    return _pos;
}

void Cluster::setCenterPosition (QVector2D pos, bool updateTransform)
{
    _pos = pos;
	_context->getSpaceProperties()->correctPosition(_pos);
	if (updateTransform) {
		updateTransformationMatrix();
	}
}

qreal Cluster::getAngle () const
{
    return _angle;
}

void Cluster::setAngle (qreal angle, bool updateTransform)
{
    _angle = angle;
    if( updateTransform )
        updateTransformationMatrix();
}

QVector2D Cluster::getVelocity () const
{
    return _vel;
}

void Cluster::setVelocity (QVector2D vel)
{
    _vel = vel;
}

qreal Cluster::getMass () const
{
    return _cells.size();
}

qreal Cluster::getAngularVel () const
{
    return _angularVel;
}

qreal Cluster::getAngularMass () const
{
    return _angularMass;
}

void Cluster::setAngularVel (qreal vel)
{
    _angularVel = vel;
}

void Cluster::updateTransformationMatrix ()
{
    _transform.setToIdentity();
    _transform.translate(_pos);
    _transform.rotate(_angle, 0.0, 0.0, 1.0);
}

QVector2D Cluster::relToAbsPos (QVector2D relPos) const
{
    return applyTransformation(relPos);
}

QVector2D Cluster::absToRelPos (QVector2D absPos) const
{
    return applyInverseTransformation(absPos);
}



Cell* Cluster::findNearestCell (QVector2D pos) const
{
    foreach( Cell* cell, _cells)
        if((calcPosition(cell, true)-pos).lengthSquared() < 0.5)
            return cell;
    return 0;
}

void Cluster::getConnectedComponent(Cell* cell, QList< Cell* >& component) const
{
    component.clear();
	auto tagGen = ServiceLocator::getInstance().getService<TagGenerator>();
    getConnectedComponent(cell, _context->getNumberGenerator()->getTag(), component);
}

void Cluster::getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const
{
    if( cell->getTag() != tag ) {
        cell->setTag(tag);
        component << cell;
        for( int i = 0; i < cell->getNumConnections(); ++i ) {
            getConnectedComponent(cell->getConnection(i), tag, component);
        }
    }
}

void Cluster::updateInternals(MaintainCenter maintanCenter /*= MaintainCenter::No*/)
{
	updateRelCoordinates(MaintainCenter::Yes);
	updateAngularMass();
}

void Cluster::radiation (qreal& energy, Cell* originCell, Particle*& energyParticle) const
{
	auto parameters = _context->getSimulationParameters();
	auto numberGen = _context->getNumberGenerator();
	energyParticle = 0;

    //1. step: calculate thermal radiation via power law (Stefan-Boltzmann law in 2D: Power ~ T^3)
    qreal radEnergy = qPow(energy, parameters->radiationExponent) * parameters->radiationFactor;

    //2. step: calculate radiation frequency
    qreal radFrequency = parameters->radiationProb;
/*    if( (radEnergy / radFrequency) < 1.0) {
        radFrequency = radEnergy;
    }*/

    //2. step: distribute the radiated energy to energy particles
    if(numberGen->getRandomReal() < radFrequency) {
        radEnergy = radEnergy / radFrequency;
        radEnergy = radEnergy *2.0 * numberGen->getRandomReal();
        if( radEnergy > (energy-1.0) )
            radEnergy = energy-1.0;
        energy = energy - radEnergy;

        //create energy particle with radEnergy
        QVector2D velPerturbation((numberGen->getRandomReal() - 0.5) * parameters->radiationVelocityPerturbation,
                                  (numberGen->getRandomReal() - 0.5) * parameters->radiationVelocityPerturbation);
        QVector2D posPerturbation = velPerturbation.normalized();
        EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

		QVector2D pos = calcPosition(originCell) + posPerturbation;
		QVector2D vel = originCell->getVelocity() * parameters->radiationVelocityMultiplier + velPerturbation;
		auto desc = ParticleDescription().setEnergy(radEnergy).setPos(QVector2D(pos.x(), pos.y())).setVel(QVector2D(vel.x(), vel.y()));
        energyParticle = factory->build(desc, _context);
		ParticleMetadata metadata;
		metadata.color = originCell->getMetadata().color;
        energyParticle->setMetadata(metadata);
    }
}

double Cluster::getRadius() const
{
	double result = 0.0;
	foreach(Cell* cell, _cells) {
		auto distance = cell->_relPos.length();
		if (distance > result) {
			result = distance;
		}
	}
	return result;
}

bool Cluster::connectable(Cluster* other) const
{
	auto space = _context->getSpaceProperties();
	auto parameters = _context->getSimulationParameters();

	auto distance = space->distance(_pos, other->_pos);
	if (getRadius() + other->getRadius() + distance > parameters->clusterMaxRadius) {
		return false;
	}
	return true;
}

ClusterMetadata Cluster::getMetadata() const
{
	return _meta;
}

void Cluster::setMetadata(ClusterMetadata metadata)
{
	_meta = metadata;
}

/*
    //calc rad prob via a function prob f(x) = 1-a/(x+b) with f(0)=CELL_RAD_PROB_LOW and f(1000)=CELL_RAD_PROB_HIGH
    qreal radProb = (energy * (simulationParameters.CELL_RAD_PROB_HIGH - simulationParameters.CELL_RAD_PROB_LOW) + 1000.0*simulationParameters.CELL_RAD_PROB_LOW*(1.0-simulationParameters.CELL_RAD_PROB_HIGH))
                       / (energy * (simulationParameters.CELL_RAD_PROB_HIGH - simulationParameters.CELL_RAD_PROB_LOW) + 1000.0*(1.0-simulationParameters.CELL_RAD_PROB_HIGH));
    if( (qreal)qrand()/RAND_MAX < radProb ) {

        //radiation energy between 0 and CELL_RAD_ENERGY
        qreal radEnergy = qMin(energy*simulationParameters.RAD_ENERGY_FRACTION*((qreal)qrand()/RAND_MAX)+0.0, energy-0.0);
//        qreal radEnergy = qMin(CELL_RAD_ENERGY*((qreal)qrand()/RAND_MAX)+1.0, energy-1.0);
//        if( (radEnergy >= 1.0) &&  (radEnergy < (energy-1.0) ) ) {
            energy = energy - radEnergy;

            //create energy particle with radEnergy
            QVector2D velPerturbation(((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB,
                                      ((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB, 0.0);
            QVector2D posPerturbation = velPerturbation.normalized();
            energyParticle = new EnergyParticle(radEnergy,
                                             calcPosition(originCell, cellMap)+posPerturbation,
                                             originCell->getVel()*simulationParameters.CELL_RAD_ENERGY_VEL_MULT+velPerturbation);
            energyParticle->color = originCell->getColor();
//        }
    }
*/






/*
    //apply gravitational forces
    QVector2D gSource1(200.0+qSin(0.5*degToRad*(qreal)time)*50, 200.0+qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    QVector2D gSource2(200.0-qSin(0.5*degToRad*(qreal)time)*50, 200.0-qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    foreach( Cell* cell, _cells) {
        QVector2D distance1 = gSource1-calcPosition(cell);
        QVector2D distance2 = gSource1-(calcPosition(cell)+cell->getVel());
        cellMap->correctDistance(distance1);
        cellMap->correctDistance(distance2);
        cell->_vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
        cell->_vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
        distance1 = gSource2-calcPosition(cell);
        distance2 = gSource2-(calcPosition(cell)+cell->_vel);
        cellMap->correctDistance(distance1);
        cellMap->correctDistance(distance2);
        cell->_vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
        cell->_vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
    }

    //update velocity and angular velocity
    updateVel_angularVel_via_cellVelocities();
*/

