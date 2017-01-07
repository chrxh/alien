#include "cellclusterimpl.h"

#include "model/entities/cell.h"
#include "model/factoryfacade.h"
#include "model/features/cellfeature.h"
#include "model/entities/entityfactory.h"
#include "model/entities/token.h"
#include "model/entities/energyparticle.h"
#include "model/physics/physics.h"
#include "model/config.h"
#include "model/simulationcontext.h"
#include "model/cellmap.h"
#include "model/topology.h"
#include "global/global.h"
#include "global/servicelocator.h"

#include <QMatrix4x4>
#include <qmath.h>

const int PROTECTION_COUNTER_AFTER_COLLISION = 14;

CellClusterImpl::CellClusterImpl (SimulationContext* context)
    : _context(context)
    ,  _topology(context->getTopology())
    ,  _cellMap(context->getCellMap())
    ,  _id(GlobalFunctions::createNewTag())
    ,  _color(GlobalFunctions::createNewTag())
{
    updateTransformationMatrix();
}

CellClusterImpl::CellClusterImpl(QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
    , QVector3D vel, SimulationContext* context)
    : _context(context)
    ,  _topology(context->getTopology())
    ,  _cellMap(context->getCellMap())
    ,  _angle(angle)
    ,  _pos(pos)
    ,  _angularVel(angularVel)
    ,  _vel(vel)
    ,  _cells(cells)
    ,  _id(GlobalFunctions::createNewTag())
    ,  _color(GlobalFunctions::createNewTag())
{
    _topology->correctPosition(_pos);
    foreach(Cell* cell, _cells) {
        cell->setCluster(this);
    }
    updateTransformationMatrix();
    updateRelCoordinates();
    updateAngularMass();
}

CellClusterImpl::CellClusterImpl(QList< Cell* > cells, qreal angle, SimulationContext* context)
    : _context(context)
    ,  _topology(context->getTopology())
    ,  _cellMap(context->getCellMap())
    ,  _angle(angle)
    ,  _cells(cells)
    ,  _id(GlobalFunctions::createNewTag())
    ,  _color(GlobalFunctions::createNewTag())
{
    //calc new center
    QVector3D center;
    foreach( Cell* cell, _cells) {
        center += cell->getCluster()->calcPosition(cell);
    }
    center /= _cells.size();
    setPosition(center);

    //set rel coordinated with respect to the new center
    foreach(Cell* cell, _cells) {

        //adjust relative position of the cells
        QVector3D pos(cell->getCluster()->calcPosition(cell));
        cell->setCluster(this);
        cell->setAbsPosition(pos);
    }

    updateAngularMass();
    updateVel_angularVel_via_cellVelocities();
}

CellClusterImpl::~CellClusterImpl ()
{
    foreach(Cell* cell, _cells) {
        delete cell;
    }
}

void CellClusterImpl::clearCellsFromMap ()
{
    foreach( Cell* cell, _cells) {
        _cellMap->removeCellIfPresent(_transform.map(cell->getRelPos()), cell);
    }
}

void CellClusterImpl::clearCellFromMap (Cell* cell)
{
    _cellMap->removeCellIfPresent(_transform.map(cell->getRelPos()), cell);
}

void CellClusterImpl::drawCellsToMap ()
{
    foreach( Cell* cell, _cells) {
        QVector3D pos(calcPosition(cell, true));
        _cellMap->setCell(pos, cell);
    }
}

//step 1:
//initiate movement of particles
void CellClusterImpl::processingInit ()
{
    //clear cells
    foreach( Cell* cell, _cells) {

        //remove particle from old position
        //-> note that due to numerical effect during fusion position can be slightly changed
        _cellMap->removeCell(_transform.map(cell->getRelPos()));
        if( cell->getProtectionCounter() > 0 )
            cell->setProtectionCounter(cell->getProtectionCounter()-1);
    }
}

//step 2: dissipation, returns lost energy
void CellClusterImpl::processingDissipation (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles)
{
    updateCellVel();

    //determine energies (the new kinetic energy will be calculated later)
    qreal oldEnergy = Physics::kineticEnergy(_cells.size(), _vel, _angularMass, _angularVel);
    qreal newEnergy = 0.0;

    //dissipation
    bool cellDestroyed(false);
    QMutableListIterator<Cell*> i(_cells);
    while (i.hasNext()) {
        Cell* cell(i.next());
        EnergyParticle* energyParticle(0);

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
                radiation(cell->getToken(i)->energy, cell, energyParticle);
                if( energyParticle )
                    energyParticles << energyParticle;
            }
        }

        //kill cell?
        if( (cell->isToBeKilled() || (cell->getEnergy() < simulationParameters.CRIT_CELL_TRANSFORM_ENERGY)) ) {
            qreal kinEnergy = Physics::kineticEnergy(1.0, cell->getVel(), 0.0, 0.0);
            qreal internalEnergy = cell->getEnergyIncludingTokens();
            EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
            qreal energyForParticle = internalEnergy
                + kinEnergy / simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
            energyParticle = factory->buildEnergyParticle(
                energyForParticle, calcPosition(cell, true), cell->getVel(), _context);
            energyParticle->color = cell->getColor();
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
            quint64 tag(GlobalFunctions::createNewTag());
            getConnectedComponent(_cells[0], tag, component);
            if( component.size() < size ) {
                EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
                CellCluster* part = factory->buildCellClusterFromForeignCells(
                    component, _angle, _context);
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
                qreal diffEnergyCell = (diffEnergy/static_cast<qreal>(size))/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
                foreach(Cell* cell, _cells) {
                    if( cell->getEnergy() > (-diffEnergyCell) )
                        cell->setEnergy(cell->getEnergy() + diffEnergyCell);
                }
                return;
            }
        }
        while( !_cells.isEmpty() );

        //calc energy difference
        foreach(CellCluster* cluster, fragments) {
            newEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(), cluster->getVel(), cluster->getAngularMass(), cluster->getAngularVel());
        }
        qreal diffEnergy = oldEnergy-newEnergy;

        //spread energy difference on cells
        qreal diffEnergyCell = (diffEnergy/static_cast<qreal>(size))/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
        foreach(CellCluster* cluster, fragments)
            foreach(Cell* cell, cluster->getCellsRef()) {
                if( cell->getEnergy() > (-diffEnergyCell) )
                    cell->setEnergy(cell->getEnergy() + diffEnergyCell);
            }

        //largest CellClusterImpl inherits the color
        int largestSize(0);
        CellCluster* largestCluster(0);
        foreach( CellCluster* cluster, fragments) {
            if( cluster->getCellsRef().size() > largestSize ) {
                largestSize = cluster->getCellsRef().size();
                largestCluster = cluster;
            }
        }
        largestCluster->setColor(_color);
    }
}

//step 3:
//actual movement, fusion and collision
//set cells to cellMap map
void CellClusterImpl::processingMovement ()
{
    struct CollisionData {
        int movementState = 0;  //0: will do nothing, 1: collision, 2: fusion
        CellSet overlappingCells;
        QList< QPair< Cell*, Cell* > > overlappingCellPairs;
    };

    //(gravitational force)
//    QVector3D f = QVector3D(300.0, 300.0, 0.0)-_pos;
//    _vel += f/(1.0+f.lengthSquared())*0.05;

    //add velocity
//    qreal oldAngle = _angle;
//    QVector3D oldPos = _pos;
    _angle += _angularVel;
    if( _angle > 360.0 )
        _angle -= 360.0;
    if( _angle < 0.0 )
        _angle += 360.0;
    _pos += _vel;
    _topology->correctPosition(_pos);
    updateTransformationMatrix();
    QVector3D pos;

    //collect information for every colliding cluster
    QMap< quint64, CollisionData > clusterCollisionDataMap;
    QMap< quint64, Cell* > idCellMap;
    QMap< quint64, CellCluster* > idClusterMap;
    foreach( Cell* cell, _cells) {
        pos = calcPosition(cell, true);
        for(int x = -1; x < 2; ++x)
            for(int y = -1; y < 2; ++y) {
                Cell* tempCell = _cellMap->getCell(pos+QVector3D(x, y, 0.0));
                if( tempCell )
                    if( tempCell->getCluster() != this ) {

                        //cell close enough?
                        QVector3D displacement(tempCell->getCluster()->calcPosition(tempCell, true)-pos);
                        _topology->correctDisplacement(displacement);
                        if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MAX ) {
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
                                if( cell->connectable(tempCell) && ((cell->getVel()-tempCell->getVel()).length() >= simulationParameters.CLUSTER_FUSION_VEL) )
                                    colData.movementState = 2;

                                //collision possible?
                                else if( cell->getProtectionCounter() == 0 && tempCell->getProtectionCounter() == 0 )
                                    colData.movementState = 1;

                            }

                            //cluster already set for collision?
                            if( colData.movementState == 1 ) {

                                //fusion possible?
                                if( cell->connectable(tempCell) && ((cell->getVel()-tempCell->getVel()).length() >= simulationParameters.CLUSTER_FUSION_VEL) )
                                    colData.movementState = 2;
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
        CellCluster* otherCluster = idClusterMap[it.key()];
        CollisionData collisionData = it.value();


        //taking fusion probability into account
/*        if( (colData.movementState == 2) && ((qreal)qrand()/RAND_MAX > simulationParameters.CLUSTER_FUSION_PROB) ) {
            colData.movementState = 1;
        }
*/
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
            QVector3D centerPos(0.0, 0.0, 0.0);
            for (Cell* otherCell : collisionData.overlappingCells) {
                centerPos = centerPos + otherCluster->calcPosition(otherCell);
            }
            centerPos = centerPos/collisionData.overlappingCells.size();

            //calc negative velocity at the center position (later used as outerSpace vector)
            QVector3D rAPp = centerPos-_pos;
            _topology->correctDisplacement(rAPp);
            rAPp = Physics::rotateQuarterCounterClockwise(rAPp);
            QVector3D rBPp = centerPos-otherCluster->getPosition();
            _topology->correctDisplacement(rBPp);
            rBPp = Physics::rotateQuarterCounterClockwise(rBPp);
            QVector3D outerSpace = (otherCluster->getVel()-rBPp*otherCluster->getAngularVel()*degToRad)-(_vel-rAPp*_angularVel*degToRad);

            //calc center normal vector of the overlapping cells from the other cluster
            QVector3D n;
            for (Cell* otherCell : collisionData.overlappingCells) {
                n = n + otherCell->calcNormal(outerSpace).normalized();
            }

            //calc new vectors
            qreal mA = _cells.size();
            qreal mB = otherCluster->getCellsRef().size();
            QVector3D vA2, vB2;
            qreal angularVelA2 = 0;
            qreal angularVelB2 = 0;
            n.normalize();
            if( n.length() < ALIEN_PRECISION )
                n.setX(1.0);

            Physics::collision(_vel, otherCluster->getVel(),//, clusterPos, otherClusterPos, centerPos,
                               rAPp, rBPp,
                               _angularVel, otherCluster->getAngularVel(), n,
                               _angularMass, otherCluster->getAngularMass(), mA, mB, vA2, vB2, angularVelA2,
                               angularVelB2);

            //set new vectors
            _vel = vA2;
            otherCluster->setVel(vB2);
            _angularVel = angularVelA2;
            otherCluster->setAngularVel(angularVelB2);

            //add new velocities to the old positions
//            _angle = oldAngle+_angularVel;
//            _pos = oldPos+_vel;
//            cellMap->correctPosition(_pos);
//            calcTransform();
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
                QVector3D displacement(otherCell->getCluster()->calcPosition(otherCell, true)-calcPosition(cell, true));
                _topology->correctDisplacement(displacement);

                //kill cell if too close
                if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MIN ){
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
                qreal eKinOld2 = Physics::kineticEnergy(mB, otherCluster->getVel(), otherCluster->getAngularMass(), otherCluster->getAngularVel());

                if( otherCluster->getCellsRef().size() > _cells.size() )
                    _color = otherCluster->getColor();

//                qDebug("cluster center: (%f, %f)",_pos.x(), _pos.y);

                //calculate new center
                QVector3D centre(0.0,0.0,0.0);
                QVector3D correction(_topology->correctionIncrement(_pos, otherCluster->getPosition()));
                foreach( Cell* cell, _cells) {
                    cell->setRelPos(calcPosition(cell));     //store absolute position only temporarily
                    centre += cell->getRelPos();
                }
                foreach( Cell* cell, otherCluster->getCellsRef()) {
                    cell->setRelPos(otherCluster->calcPosition(cell)+correction);
                    centre += cell->getRelPos();
                }
                centre /= (_cells.size()+otherCluster->getCellsRef().size());
                _pos = centre;
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
                    cell->setRelPos(absToRelPos(cell->getRelPos()));
                }
                _topology->correctPosition(_pos);
                updateTransformationMatrix();

                //calc angular mass, velocity, angular velocity
                updateAngularMass();
                updateVel_angularVel_via_cellVelocities();

                //calc newkinetic energy of united cluster
                qreal eKinNew = Physics::kineticEnergy(_cells.size(), _vel, _angularMass, _angularVel);

                //spread lost kinetic energy to tokens and internal energy of the fused cells
                qreal eDiff = ((eKinOld1 + eKinOld2 - eKinNew)/static_cast<qreal>(fusedCells.size()))/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
                if( eDiff > ALIEN_PRECISION ) {
                    for (Cell* cell : fusedCells) {

                        //create token?
                        if( (cell->getNumToken() < simulationParameters.CELL_TOKENSTACKSIZE) && (eDiff > simulationParameters.MIN_TOKEN_ENERGY) ) {
                            Token* token = new Token(eDiff, true);
                            cell->addToken(token, Cell::ACTIVATE_TOKEN::NOW, Cell::UPDATE_TOKEN_ACCESS_NUMBER::YES);
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
        QVector3D pos = _transform.map(cell->getRelPos());
        _cellMap->setCell(pos, cell);
    }

}

//step 4:
//token processing
void CellClusterImpl::processingToken (QList< EnergyParticle* >& energyParticles, bool& decompose)
{
    std::vector<Token*> spreadToken(simulationParameters.MAX_CELL_CONNECTIONS);
    std::vector<Cell*> spreadTokenCells(simulationParameters.MAX_CELL_CONNECTIONS);

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
                if( (((tokenAccessNumber+1)%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber()) && (!otherCell->isTokenBlocked())
                    && (otherCell->getNumToken(true) < simulationParameters.CELL_TOKENSTACKSIZE ) ) {
                    ++numPlaces;
                }
            }

            //no free places for token?
            if( numPlaces == 0 ) {
                cell->setEnergy(cell->getEnergy() + token->energy);
                delete token;
            }

            //free places for tokens available
            else {
                //not enough cell energy available?
                if( //(cell->_energy < ((qreal)numPlaces-1.0)*token->energy) ||
                    token->energy < simulationParameters.MIN_TOKEN_ENERGY) {
                    cell->setEnergy(cell->getEnergy() + token->energy);
                    delete token;
                }
                else {
                    //calc available token energy
                    qreal tokenEnergy = token->energy;
                    qreal availableTokenEnergy = tokenEnergy / numPlaces;

                    //spread token to free places on adjacent cells and duplicate token if necessary
                    int spreadTokenCounter = 0;
                    for(int j = 0; j < cell->getNumConnections(); ++j) {
                        Cell* otherCell = cell->getConnection(j);
                        if( (((tokenAccessNumber+1)%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber()) && (!otherCell->isTokenBlocked())
                            && (otherCell->getNumToken(true) < simulationParameters.CELL_TOKENSTACKSIZE ) ) {
                            if( spreadTokenCounter > 0 ) {
                                spreadTokenCells[spreadTokenCounter] = otherCell;
                                spreadToken[spreadTokenCounter] = token->duplicate();
                                otherCell->addToken(spreadToken[spreadTokenCounter], Cell::ACTIVATE_TOKEN::LATER, Cell::UPDATE_TOKEN_ACCESS_NUMBER::YES);
                            }
                            if( spreadTokenCounter == 0 ) {
                                spreadTokenCells[0] = otherCell;
                                spreadToken[0] = token;
                                otherCell->addToken(token, Cell::ACTIVATE_TOKEN::LATER, Cell::UPDATE_TOKEN_ACCESS_NUMBER::YES);
                            }
                            if( numPlaces > 1 ) {
                                spreadToken[spreadTokenCounter]->energy = availableTokenEnergy;

                                //transfer remaining energy from cell to token if possible
                                if( otherCell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+tokenEnergy-availableTokenEnergy) ) {
                                    spreadToken[spreadTokenCounter]->energy = tokenEnergy;
                                    otherCell->setEnergy(otherCell->getEnergy() - (tokenEnergy-availableTokenEnergy));
                                }
                                else if( otherCell->getEnergy() > simulationParameters.CRIT_CELL_TRANSFORM_ENERGY ) {
                                    spreadToken[spreadTokenCounter]->energy += otherCell->getEnergy() - simulationParameters.CRIT_CELL_TRANSFORM_ENERGY;
                                    otherCell->setEnergy(simulationParameters.CRIT_CELL_TRANSFORM_ENERGY);
                                }
                            }
                            spreadTokenCounter++;
                        }
                    }


                    //execute cell functions and token energy guidance system on the cells with the tokens
                    for( int i = 0; i < spreadTokenCounter; ++i ) {

                        //execute cell function
                        CellFeature::ProcessingResult processingResult = spreadTokenCells[i]->getFeatures()->process(spreadToken[i], spreadTokenCells[i], cell);
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

//step 5:
//activate new token and kill cells which are too close or where too much forces are applied
void CellClusterImpl::processingFinish ()
{
    qreal maxClusterRadius = qMin(_topology->getSize().x/2.0, _topology->getSize().y/2.0);
    foreach( Cell* cell, _cells) {

        //activate tokens
        cell->activatingNewTokens();

        //kill cells which are too far from cluster center
        if(cell->getRelPos().length() > (maxClusterRadius-1.0) )
            cell->setToBeKilled(true);

        //find nearby cells and kill if they are too close
        QVector3D pos = calcPosition(cell, true);
        for( int x = -1; x < 2; ++x )
            for( int y = -1; y < 2; ++y ) {
                Cell* otherCell(_cellMap->getCell(pos+QVector3D(x, y, 0.0)));
                if( otherCell ) {
                    if( otherCell != cell ) {
//                    if( otherCell->_cluster != this ) {
                        CellCluster* otherCluster = otherCell->getCluster();
//                        foreach(Cell* otherCell2, otherCluster->getCellsRef()) {
//                            if( otherCell2 != cell ) {
                                QVector3D displacement = otherCluster->calcPosition(otherCell, true)-calcPosition(cell, true);
                                _topology->correctDisplacement(displacement);
                                if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MIN ){
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

void CellClusterImpl::addCell (Cell* cell, QVector3D absPos)
{
    cell->setRelPos(absToRelPos(absPos));
    cell->setCluster(this);
    _cells << cell;

    updateRelCoordinates(true);
    updateAngularMass();
}

void CellClusterImpl::removeCell (Cell* cell, bool maintainCenter)
{
    cell->delAllConnection();
    _cells.removeAll(cell);

    updateRelCoordinates(maintainCenter);
    updateAngularMass();
}

void CellClusterImpl::updateCellVel (bool forceCheck)
{
    if( _cells.size() == 1 ) {
        _cells[0]->setVel(_vel);
    }
    else {

        //calc cell velocities
        foreach( Cell* cell, _cells) {
            QVector3D vel = Physics::tangentialVelocity(calcCellDistWithoutTorusCorrection(cell), _vel, _angularVel);
            if( cell->getVel().isNull() ) {
                cell->setVel(vel);
            }
            else {
                QVector3D a = vel - cell->getVel();

                //destroy cell if acceleration exceeds a certain threshold
                if( forceCheck ) {
                    if( a.length() > simulationParameters.CELL_MAX_FORCE ) {
                        if(GlobalFunctions::random() < simulationParameters.CELL_MAX_FORCE_DECAY_PROB )
                            cell->setToBeKilled(true);
                    }
                }
                cell->setVel(vel);
            }
        }
    }
}

void CellClusterImpl::updateAngularMass () {

    //calc angular mass
    _angularMass = 0.0;
    foreach( Cell* cell, _cells)
        _angularMass += (cell->getRelPos().lengthSquared());
}

void CellClusterImpl::updateRelCoordinates (bool maintainCenter)
{
    if( maintainCenter ) {

        //calc new center in relative coordinates
//        calcTransform();
        QVector3D center(0.0,0.0,0.0);
        foreach( Cell* cell, _cells) {
            center += cell->getRelPos();
        }
        center /= _cells.size();

        //set rel coordinated with respect to the new center
        foreach( Cell* cell, _cells) {
            cell->setRelPos(cell->getRelPos() - center);
        }
    }
    else {

        //calc new center
//        calcTransform();
        QVector3D centre(0.0,0.0,0.0);
        foreach( Cell* cell, _cells) {
            centre += _transform.map(cell->getRelPos());
        }
        centre /= _cells.size();

        //center transformation
        QMatrix4x4 oldTransform(_transform);
        _pos = centre;
        updateTransformationMatrix();
        QMatrix4x4 newTransformInv(_transform.inverted());

        //set rel coordinated with respect to the new center
        foreach( Cell* cell, _cells) {
            cell->setRelPos(newTransformInv.map(oldTransform.map(cell->getRelPos())));
        }
    }
}

//Note: angular mass needs to be calculated before, energy may be lost
void CellClusterImpl::updateVel_angularVel_via_cellVelocities ()
{
    if( _cells.size() > 1 ) {

        //first step: calc cluster mean velocity
        _vel = QVector3D(0.0, 0.0, 0.0);
        foreach( Cell* cell, _cells ) {
            _vel += cell->getVel();
        }
        _vel = _vel/_cells.size();

        //second step: calc angular momentum for the cluster in the inertia system with velocity _vel
        qreal angularMomentum = 0.0;
        foreach( Cell* cell, _cells ) {
            QVector3D r = calcPosition(cell)-_pos;
            QVector3D v = cell->getVel() - _vel;
            angularMomentum += Physics::angularMomentum(r, v);     //we only need the 3rd component of the 3D cross product
        }

        //third step: calc angular velocity via the third component of the angular momentum
        _angularVel = Physics::angularVelocity(angularMomentum, _angularMass);

    }
    else if( _cells.size() == 1 ) {
        _vel = _cells[0]->getVel();
        _angularVel = 0.0;
    }
}


QVector3D CellClusterImpl::calcPosition (const Cell* cell, bool topologyCorrection) const
{
    QVector3D cellPos(_transform.map(cell->getRelPos()));
    if(  topologyCorrection )
        _topology->correctPosition(cellPos);
    return cellPos;
}

QVector3D CellClusterImpl::calcCellDistWithoutTorusCorrection (Cell* cell) const
{
    return calcPosition(cell)-_pos;
}

QList< CellCluster* > CellClusterImpl::decompose () const
{
    QList< CellCluster* > fragments;
    while( !_cells.isEmpty() ) {


        //find fragment
        QList< Cell* > component;
        quint64 tag(GlobalFunctions::createNewTag());
        getConnectedComponent(_cells[0], tag, component);

        //remove fragment from clusters
        QMap< quint64, CellCluster* > idClusterMap;
        foreach( Cell* cell, component) {
            idClusterMap[cell->getCluster()->getId()] = cell->getCluster();
        }

        foreach( CellCluster* cluster, idClusterMap.values() ) {
            QMutableListIterator<Cell*> i(cluster->getCellsRef());
            while (i.hasNext()) {
                if( i.next()->getTag() == tag )
                    i.remove();
            }
        }
        fragments << new CellClusterImpl(component, _angle, _context);
    }
    return fragments;
}

qreal CellClusterImpl::calcAngularMassWithNewParticle (QVector3D particlePos) const
{

    //calc new center
    QVector3D particleRelPos = absToRelPos(particlePos);
    QVector3D center = particleRelPos;
    foreach(Cell* cell, _cells) {
        center = center + cell->getRelPos();
    }
    center = center / (_cells.size()+1);

    //calc new angular mass
    QVector3D diff = particleRelPos - center;
    _topology->correctDisplacement(diff);
    qreal aMass = diff.lengthSquared();
    foreach(Cell* cell, _cells) {
        diff = cell->getRelPos() - center;
        _topology->correctDisplacement(diff);
        aMass = aMass + diff.lengthSquared();
    }
    return aMass;
}

qreal CellClusterImpl::calcAngularMassWithoutUpdate () const
{

    //calc new center
    QVector3D center(0.0, 0.0, 0.0);
    foreach(Cell* cell, _cells) {
        center = center + cell->getRelPos();
    }
    center = center / (_cells.size());

    //calc new angular mass
    qreal aMass = 0.0;
    foreach(Cell* cell, _cells) {
        QVector3D displacement = cell->getRelPos() - center;
        _topology->correctDisplacement(displacement);
        aMass = aMass + displacement.lengthSquared();
    }
    return aMass;
}

bool CellClusterImpl::isEmpty() const
{
    return _cells.isEmpty();
}

QList< Cell* >& CellClusterImpl::getCellsRef ()
{
    return _cells;
}

/*QVector3D CellClusterImpl::getCoordinate (Cell* cell)
{
    return _transform.map(cell->getRelPos());
}
*/

void CellClusterImpl::findNearestCells (QVector3D pos, Cell*& cell1, Cell*& cell2) const
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

const quint64& CellClusterImpl::getId () const
{
    return _id;
}

void CellClusterImpl::setId (quint64 id)
{
    _id = id;
}

QList< quint64 > CellClusterImpl::getCellIds () const
{
    QList< quint64 > ids;
    foreach( Cell* cell, _cells )
        ids << cell->getId();
    return ids;
}

quint64 CellClusterImpl::getColor () const
{
    return _color;
}

void CellClusterImpl::setColor (quint64 color)
{
    _color = color;
}

QVector3D CellClusterImpl::getPosition () const
{
    return _pos;
}

void CellClusterImpl::setPosition (QVector3D pos, bool updateTransform)
{
    _pos = pos;
    if( updateTransform )
        updateTransformationMatrix();
}

qreal CellClusterImpl::getAngle () const
{
    return _angle;
}

void CellClusterImpl::setAngle (qreal angle, bool updateTransform)
{
    _angle = angle;
    if( updateTransform )
        updateTransformationMatrix();
}

QVector3D CellClusterImpl::getVel () const
{
    return _vel;
}

void CellClusterImpl::setVel (QVector3D vel)
{
    _vel = vel;
}

qreal CellClusterImpl::getMass () const
{
    return _cells.size();
}

qreal CellClusterImpl::getAngularVel () const
{
    return _angularVel;
}

qreal CellClusterImpl::getAngularMass () const
{
    return _angularMass;
}

void CellClusterImpl::setAngularVel (qreal vel)
{
    _angularVel = vel;
}

void CellClusterImpl::updateTransformationMatrix ()
{
    _transform.setToIdentity();
    _transform.translate(_pos);
    _transform.rotate(_angle, 0.0, 0.0, 1.0);
}

QVector3D CellClusterImpl::relToAbsPos (QVector3D relPos) const
{
    return _transform.map(relPos);
}

QVector3D CellClusterImpl::absToRelPos (QVector3D absPos) const
{
    return _transform.inverted().map(absPos);
}



Cell* CellClusterImpl::findNearestCell (QVector3D pos) const
{
    foreach( Cell* cell, _cells)
        if((calcPosition(cell, true)-pos).lengthSquared() < 0.5)
            return cell;
    return 0;
}

void CellClusterImpl::getConnectedComponent(Cell* cell, QList< Cell* >& component) const
{
    component.clear();
    quint64 tag(GlobalFunctions::createNewTag());
    getConnectedComponent(cell, tag, component);
}

void CellClusterImpl::getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const
{
    if( cell->getTag() != tag ) {
        cell->setTag(tag);
        component << cell;
        for( int i = 0; i < cell->getNumConnections(); ++i ) {
            getConnectedComponent(cell->getConnection(i), tag, component);
        }
    }
}

void CellClusterImpl::radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle) const
{
    energyParticle = 0;

    //1. step: calculate thermal radiation via power law (Stefan-Boltzmann law in 2D: Power ~ T^3)
    qreal radEnergy = qPow(energy, simulationParameters.RAD_EXPONENT) * simulationParameters.RAD_FACTOR;

    //2. step: calculate radiation frequency
    qreal radFrequency = simulationParameters.RAD_PROBABILITY;
/*    if( (radEnergy / radFrequency) < 1.0) {
        radFrequency = radEnergy;
    }*/

    //2. step: distribute the radiated energy to energy particles
    if(GlobalFunctions::random() < radFrequency) {
        radEnergy = radEnergy / radFrequency;
        radEnergy = radEnergy *2.0 * (static_cast<qreal>(qrand()) /RAND_MAX);
        if( radEnergy > (energy-1.0) )
            radEnergy = energy-1.0;
        energy = energy - radEnergy;

        //create energy particle with radEnergy
        QVector3D velPerturbation((GlobalFunctions::random() -0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB,
                                  (GlobalFunctions::random() -0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB, 0.0);
        QVector3D posPerturbation = velPerturbation.normalized();
        EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
        energyParticle = factory->buildEnergyParticle(radEnergy
            , calcPosition(originCell)+posPerturbation
            , originCell->getVel()*simulationParameters.CELL_RAD_ENERGY_VEL_MULT+velPerturbation
            , _context);
        energyParticle->color = originCell->getColor();
    }
}

void CellClusterImpl::serializePrimitives (QDataStream& stream) const
{
    stream << _angle << _pos << _angularVel << _vel;
    /*stream << _cells.size();
    FactoryFacade *facade = ServiceLocator::getInstance().getService<FactoryFacade>();
    foreach( Cell* cell, _cells ) {
        facade->serializeFeaturedCell(cell, stream);
    }*/
    stream << _id << _color;
}

void CellClusterImpl::deserializePrimitives(QDataStream& stream)
{
	stream >> _angle >> _pos >> _angularVel >> _vel;
	stream >> _id >> _color;
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
            QVector3D velPerturbation(((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB,
                                      ((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB, 0.0);
            QVector3D posPerturbation = velPerturbation.normalized();
            energyParticle = new EnergyParticle(radEnergy,
                                             calcPosition(originCell, cellMap)+posPerturbation,
                                             originCell->getVel()*simulationParameters.CELL_RAD_ENERGY_VEL_MULT+velPerturbation);
            energyParticle->color = originCell->getColor();
//        }
    }
*/






/*
    //apply gravitational forces
    QVector3D gSource1(200.0+qSin(0.5*degToRad*(qreal)time)*50, 200.0+qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    QVector3D gSource2(200.0-qSin(0.5*degToRad*(qreal)time)*50, 200.0-qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    foreach( Cell* cell, _cells) {
        QVector3D distance1 = gSource1-calcPosition(cell);
        QVector3D distance2 = gSource1-(calcPosition(cell)+cell->getVel());
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


