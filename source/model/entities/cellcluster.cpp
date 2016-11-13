#include "cellcluster.h"

#include "model/modelfacade.h"
#include "model/features/cellfeature.h"
#include "model/entities/entityfactory.h"
#include "model/entities/token.h"
#include "model/physics/physics.h"
#include "model/simulationsettings.h"
#include "global/global.h"
#include "global/servicelocator.h"

#include <QMatrix4x4>
#include <qmath.h>

const int PROTECTION_COUNTER_AFTER_COLLISION = 14;

CellCluster* CellCluster::buildEmptyCellCluster (Grid*& grid)
{
    return new CellCluster(grid);
}

CellCluster* CellCluster::buildCellCluster (QList< Cell* > cells,
                                           qreal angle,
                                           QVector3D pos,
                                           qreal angularVel,
                                           QVector3D vel,
                                           Grid*& grid)
{
    return new CellCluster(cells, angle, pos, angularVel, vel, grid);
}

CellCluster* CellCluster::buildCellCluster (QDataStream& stream,
                                           QMap< quint64, quint64 >& oldNewClusterIdMap,
                                           QMap< quint64, quint64 >& oldNewCellIdMap,
                                           QMap< quint64, Cell* >& oldIdCellMap,
                                           Grid*& grid)
{
    return new CellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, grid);
}

CellCluster* CellCluster::buildCellClusterFromForeignCells (QList< Cell* > cells,
                                                           qreal angle,
                                                           Grid*& grid)
{
    return new CellCluster(cells, angle, grid);
}

CellCluster::~CellCluster ()
{
    foreach(Cell* cell, _cells) {
        delete cell;
    }
}

void CellCluster::clearCellsFromMap ()
{
    foreach( Cell* cell, _cells) {
        _grid->removeCellIfPresent(_transform.map(cell->getRelPos()), cell);
    }
}

void CellCluster::clearCellFromMap (Cell* cell)
{
    _grid->removeCellIfPresent(_transform.map(cell->getRelPos()), cell);
}

void CellCluster::drawCellsToMap ()
{
    foreach( Cell* cell, _cells) {
        QVector3D pos(calcPosition(cell, true));
        _grid->setCell(pos, cell);
    }
}

//step 1:
//initiate movement of particles
void CellCluster::movementProcessingStep1 ()
{
    //clear cells
    foreach( Cell* cell, _cells) {

        //remove particle from old position
        //-> note that due to numerical effect during fusion position can be slightly changed
        _grid->removeCell(_transform.map(cell->getRelPos()));
        if( cell->getProtectionCounter() > 0 )
            cell->setProtectionCounter(cell->getProtectionCounter()-1);
    }
}

//step 2: dissipation, returns lost energy
void CellCluster::movementProcessingStep2 (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles)
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
            energyParticle =  new EnergyParticle(internalEnergy + kinEnergy / simulationParameters.INTERNAL_TO_KINETIC_ENERGY,
                                              calcPosition(cell, true),
                                              cell->getVel(), _grid);
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
            quint64 tag(GlobalFunctions::getTag());
            getConnectedComponent(_cells[0], tag, component);
            if( component.size() < size ) {
                CellCluster* part(new CellCluster(component, _angle, _grid));
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
//                if( qAbs(diffEnergy) > 10.0 )
//                    qDebug("ups1 %f, size: %d, pos x: %f, pos y: %f, , oldpos x: %f, oldpos y: %f",diffEnergy,  _cells.size(), _pos.x(), _pos.y(), oldPos.x(), oldPos.y());

                //spread energy difference on cells
                qreal diffEnergyCell = (diffEnergy/(qreal)size)/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
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
            newEnergy += Physics::kineticEnergy(cluster->_cells.size(), cluster->_vel, cluster->_angularMass, cluster->_angularVel);
        }
        qreal diffEnergy = oldEnergy-newEnergy;
//        if( qAbs(diffEnergy) > 10.0 ) {
//            qDebug("ups2 %f", diffEnergy);
//            foreach(CellCluster* cluster, fragments) {
//                qDebug("size: %d, pos x: %f, pos y: %f", cluster->_cells.size(), cluster->_cells.size(), cluster->_pos.x(), cluster->_pos.y());
//            }
//        }

        //spread energy difference on cells
        qreal diffEnergyCell = (diffEnergy/(qreal)size)/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
        foreach(CellCluster* cluster, fragments)
            foreach(Cell* cell, cluster->_cells) {
                if( cell->getEnergy() > (-diffEnergyCell) )
                    cell->setEnergy(cell->getEnergy() + diffEnergyCell);
            }

        //largest cellcluster inherits the color
        int largestSize(0);
        CellCluster* largestCluster(0);
        foreach( CellCluster* cluster, fragments) {
            if( cluster->_cells.size() > largestSize ) {
                largestSize = cluster->_cells.size();
                largestCluster = cluster;
            }
        }
        largestCluster->_color = _color;
    }
}

//step 3:
//actual movement, fusion and collision
//set cells to grid map
void CellCluster::movementProcessingStep3 ()
{
    struct CollisionData {
        int movementState;  //0: will do nothing, 1: collision, 2: fusion
        QSet< quint64 > overlappingCells;
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
    _grid->correctPosition(_pos);
    calcTransform();
    QVector3D pos;

    //collect information for every colliding cluster
    QMap< quint64, CollisionData > clusterCollisionDataMap;
    QMap< quint64, Cell* > idCellMap;
    QMap< quint64, CellCluster* > idClusterMap;
    foreach( Cell* cell, _cells) {
        pos = calcPosition(cell, true);
        for(int x = -1; x < 2; ++x)
            for(int y = -1; y < 2; ++y) {
                Cell* tempCell = _grid->getCell(pos+QVector3D(x, y, 0.0));
                if( tempCell )
                    if( tempCell->getCluster() != this ) {

                        //cell close enough?
                        QVector3D displacement(tempCell->getCluster()->calcPosition(tempCell, true)-pos);
                        _grid->correctDisplacement(displacement);
                        if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MAX ) {
                            quint64 clusterId = tempCell->getCluster()->_id;

                            //read collision data for the colliding cluster
                            CollisionData colData;
                            if( clusterCollisionDataMap.contains(clusterId) )
                                colData = clusterCollisionDataMap[clusterId];
                            else
                                colData.movementState = 0;

                            //remember cell
                            idCellMap[tempCell->getId()] = tempCell;
                            idClusterMap[clusterId] = tempCell->getCluster();
                            colData.overlappingCells << tempCell->getId();
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
        CollisionData colData = it.value();


        //taking fusion probability into account
/*        if( (colData.movementState == 2) && ((qreal)qrand()/RAND_MAX > simulationParameters.CLUSTER_FUSION_PROB) ) {
            colData.movementState = 1;
        }
*/
        //collision?
        if( colData.movementState == 1 ) {

            //set protection counter to avoid collision in the next few time steps
            QListIterator< QPair< Cell*, Cell* > > it2(colData.overlappingCellPairs);
            while( it2.hasNext() ) {
                QPair< Cell*, Cell* > cellPair(it2.next());
                cellPair.first->setProtectionCounter(PROTECTION_COUNTER_AFTER_COLLISION);
                cellPair.second->setProtectionCounter(PROTECTION_COUNTER_AFTER_COLLISION);
            }

            //performing collisions:
            //calc center position of the overlapping cells
            QVector3D centerPos(0.0, 0.0, 0.0);
            QSetIterator< quint64 > it3(colData.overlappingCells);
            while( it3.hasNext() ) {
                Cell* otherCell(idCellMap[it3.next()]);
                centerPos = centerPos + otherCluster->_transform.map(otherCell->getRelPos());
            }
            centerPos = centerPos/colData.overlappingCells.size();

            //calc negative velocity at the center position (later used as outerSpace vector)
            QVector3D rAPp = centerPos-_pos;
            _grid->correctDisplacement(rAPp);
            rAPp = Physics::rotateQuarterCounterClockwise(rAPp);
            QVector3D rBPp = centerPos-otherCluster->_pos;
            _grid->correctDisplacement(rBPp);
            rBPp = Physics::rotateQuarterCounterClockwise(rBPp);
            QVector3D outerSpace = (otherCluster->_vel-rBPp*otherCluster->_angularVel*degToRad)-(_vel-rAPp*_angularVel*degToRad);

            //calc center normal vector of the overlapping cells from the other cluster
            it3 = colData.overlappingCells;
            QVector3D n(0.0, 0.0, 0.0);
            while( it3.hasNext() ) {
                Cell* otherCell(idCellMap[it3.next()]);
                n = n + otherCell->calcNormal(outerSpace, otherCluster->_transform).normalized();
            }
//            qDebug("o: %f, %f", outerSpace.x(), outerSpace.y());

            //calc new vectors
            qreal mA = _cells.size();
            qreal mB = otherCluster->_cells.size();
            QVector3D vA2, vB2;
            qreal angularVelA2, angularVelB2;
            n.normalize();
            if( (n.x() == 0.0) && (n.y() == 0.0) )
                n.setX(1.0);

            Physics::collision(_vel, otherCluster->_vel,//, clusterPos, otherClusterPos, centerPos,
                               rAPp, rBPp,
                               _angularVel, otherCluster->_angularVel, n,
                               _angularMass, otherCluster->_angularMass, mA, mB, vA2, vB2, angularVelA2,
                               angularVelB2);

            //set new vectors
            _vel = vA2;
            otherCluster->_vel = vB2;
            _angularVel = angularVelA2;
            otherCluster->_angularVel = angularVelB2;

            //add new velocities to the old positions
//            _angle = oldAngle+_angularVel;
//            _pos = oldPos+_vel;
//            grid->correctPosition(_pos);
//            calcTransform();
        }

        //fusion?
        if( colData.movementState == 2 ) {

            //connecting clusters if possible
            QSet< quint64 > fusedCells;
            QListIterator< QPair< Cell*, Cell* > > it2(colData.overlappingCellPairs);
            while (it2.hasNext()) {
                QPair< Cell*, Cell* > item(it2.next());
                Cell* cell(item.first);
                Cell* otherCell(item.second);
                QVector3D displacement(otherCell->getCluster()->calcPosition(otherCell, true)-calcPosition(cell, true));
                _grid->correctDisplacement(displacement);

                //kill cell if too close
                if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MIN ){
                    if( _cells.size() > otherCell->getCluster()->_cells.size()) {
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
                    fusedCells << cell->getId();
                    fusedCells << otherCell->getId();
                    idCellMap[cell->getId()] = cell;
                }
                otherCluster = otherCell->getCluster();
            }

            //cells connected?
            if( fusedCells.size() > 0) {
                updateCellVel();
                otherCluster->updateCellVel();

                //calc old kinetic energy of both clusters
                qreal mA = _cells.size();
                qreal mB = otherCluster->_cells.size();
                qreal eKinOld1 = Physics::kineticEnergy(mA, _vel, _angularMass, _angularVel);
                qreal eKinOld2 = Physics::kineticEnergy(mB, otherCluster->_vel, otherCluster->_angularMass, otherCluster->_angularVel);

                if( otherCluster->_cells.size() > _cells.size() )
                    _color = otherCluster->_color;

//                qDebug("cluster center: (%f, %f)",_pos.x(), _pos.y);

                //calculate new center
                QVector3D centre(0.0,0.0,0.0);
                QVector3D correction(calcTopologyCorrection(otherCluster));
                foreach( Cell* cell, _cells) {
                    cell->setRelPos(calcPosition(cell));     //store absolute position only temporarily
                    centre += cell->getRelPos();
                }
                foreach( Cell* cell, otherCluster->_cells) {
                    cell->setRelPos(otherCluster->calcPosition(cell)+correction);
                    centre += cell->getRelPos();
                }
                centre /= (_cells.size()+otherCluster->_cells.size());
                _pos = centre;
                calcTransform();

                //transfer cells
                QList< Cell* > cells(otherCluster->_cells);
                _cells << cells;
                otherCluster->_cells.clear();
                foreach( Cell* cell, cells) {
                    cell->setCluster(this);
                }

                //set relative coordinates
                foreach( Cell* cell, _cells) {
                    cell->setRelPos(absToRelPos(cell->getRelPos()));
                }
                _grid->correctPosition(_pos);
                calcTransform();

                //calc angular mass, velocity, angular velocity
                updateAngularMass();
                updateVel_angularVel_via_cellVelocities();

                //calc newkinetic energy of united cluster
                qreal eKinNew = Physics::kineticEnergy(_cells.size(), _vel, _angularMass, _angularVel);

                //spread lost kinetic energy to tokens and internal energy of the fused cells
                qreal eDiff = ((eKinOld1 + eKinOld2 - eKinNew)/(qreal)fusedCells.size())/simulationParameters.INTERNAL_TO_KINETIC_ENERGY;
                if( eDiff > ALIEN_PRECISION ) {
                    foreach( quint64 id, fusedCells ) {
                        Cell* cell = idCellMap[id];

                        //create token?
                        if( (cell->getNumToken() < simulationParameters.CELL_TOKENSTACKSIZE) && (eDiff > simulationParameters.MIN_TOKEN_ENERGY) ) {
                            Token* token = new Token(eDiff, true);
                            cell->addToken(token, true);
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
        _grid->setCell(pos, cell);
    }
}

//step 4:
//token processing
void CellCluster::movementProcessingStep4 (QList< EnergyParticle* >& energyParticles, bool& decompose)
{
    Token* spreadToken[simulationParameters.MAX_CELL_CONNECTIONS];
    Cell* spreadTokenCells[simulationParameters.MAX_CELL_CONNECTIONS];

    //placing new tokens
    foreach( Cell* cell, _cells) {
        Token* token = cell->takeTokenFromStack();
        if( token ) {
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
                //-----------
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

//                    cell->_energy -= ((qreal)numPlaces-1.0)*token->energy;

                    //spread token to free places on adjacent cells and duplicate token if necessary
                    int spreadTokenCounter = 0;
//                    token->energy = tokenEnergy;
                    for(int j = 0; j < cell->getNumConnections(); ++j) {
                        Cell* otherCell = cell->getConnection(j);
                        if( (((tokenAccessNumber+1)%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber()) && (!otherCell->isTokenBlocked())
                            && (otherCell->getNumToken(true) < simulationParameters.CELL_TOKENSTACKSIZE ) ) {
                            if( spreadTokenCounter > 0 ) {
                                spreadTokenCells[spreadTokenCounter] = otherCell;
                                spreadToken[spreadTokenCounter] = token->duplicate();
                                otherCell->addToken(spreadToken[spreadTokenCounter], false, true);
                            }
                            if( spreadTokenCounter == 0 ) {
                                spreadTokenCells[0] = otherCell;
                                spreadToken[0] = token;
                                otherCell->addToken(token, false, true);
                            }
                            if( numPlaces > 1 ) {
                                spreadToken[spreadTokenCounter]->energy = availableTokenEnergy;

                                //transfer remaining energy from cell to token if possible
                                if( otherCell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+tokenEnergy-availableTokenEnergy) ) {
                                    spreadToken[spreadTokenCounter]->energy = tokenEnergy;
                                    otherCell->setEnergy(otherCell->getEnergy() - tokenEnergy-availableTokenEnergy);
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
                        spreadToken[i]->setTokenAccessNumber(spreadTokenCells[i]->getTokenAccessNumber());
                        CellFeature::ProcessingResult processingResult = spreadTokenCells[i]->getFeatures()->process(spreadToken[i], spreadTokenCells[i], cell);
                        if( processingResult.decompose )
                            decompose = true;
                        if( processingResult.newEnergyParticle )
                            energyParticles << processingResult.newEnergyParticle;

                        //average internal energies
/*                        qreal av((cell->_energy + spreadTokenCells[i]->_energy)/2.0);
                        cell->_energy = av;
                        spreadTokenCells[i]->_energy = av;*/
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
void CellCluster::movementProcessingStep5 ()
{
    qreal maxClusterRadius = qMin(_grid->getSizeX()/2.0, _grid->getSizeY()/2.0);
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
                Cell* otherCell(_grid->getCell(pos+QVector3D(x, y, 0.0)));
                if( otherCell ) {
                    if( otherCell != cell ) {
//                    if( otherCell->_cluster != this ) {
                        CellCluster* otherCluster = otherCell->getCluster();
//                        foreach(Cell* otherCell2, otherCluster->getCells()) {
//                            if( otherCell2 != cell ) {
                                QVector3D displacement = otherCluster->calcPosition(otherCell, true)-calcPosition(cell, true);
                                _grid->correctDisplacement(displacement);
                                if( displacement.length() < simulationParameters.CRIT_CELL_DIST_MIN ){
                                    if( _cells.size() > otherCluster->_cells.size()) {
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

void CellCluster::addCell (Cell* cell, QVector3D absPos)
{
    cell->setRelPos(absToRelPos(absPos));
    cell->setCluster(this);
    _cells << cell;

    updateRelCoordinates(true);
    updateAngularMass();
}

void CellCluster::removeCell (Cell* cell, bool maintainCenter)
{
    cell->delAllConnection();
    _cells.removeAll(cell);

    updateRelCoordinates(maintainCenter);
    updateAngularMass();
}

void CellCluster::updateCellVel (bool forceCheck)
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
                        if( (qreal)qrand()/RAND_MAX < simulationParameters.CELL_MAX_FORCE_DECAY_PROB )
                            cell->setToBeKilled(true);
                    }
                }
                cell->setVel(vel);
            }
        }
    }
}

void CellCluster::updateAngularMass () {

    //calc angular mass
    _angularMass = 0.0;
    foreach( Cell* cell, _cells)
        _angularMass += (cell->getRelPos().lengthSquared());
}

void CellCluster::updateRelCoordinates (bool maintainCenter)
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
        calcTransform();
        QMatrix4x4 newTransformInv(_transform.inverted());

        //set rel coordinated with respect to the new center
        foreach( Cell* cell, _cells) {
            cell->setRelPos(newTransformInv.map(oldTransform.map(cell->getRelPos())));
        }
    }
}

//Note: angular mass needs to be calculated before, energy may be lost
void CellCluster::updateVel_angularVel_via_cellVelocities ()
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


QVector3D CellCluster::calcPosition (const Cell* cell, bool topologyCorrection) const
{
    QVector3D cellPos(_transform.map(cell->getRelPos()));
    if(  topologyCorrection )
        _grid->correctPosition(cellPos);
    return cellPos;
}

//calc correction for torus topology
QVector3D CellCluster::calcTopologyCorrection (CellCluster* cluster)
{
    QVector3D correction;
    if( (cluster->_pos.x()-_pos.x()) > (_grid->getSizeX()/2.0) )
        correction.setX(-_grid->getSizeX());
    if( (_pos.x()-cluster->_pos.x()) > (_grid->getSizeX()/2.0) )
        correction.setX(_grid->getSizeX());
    if( (cluster->_pos.y()-_pos.y()) > (_grid->getSizeY()/2.0) )
        correction.setY(-_grid->getSizeY());
    if( (_pos.y()-cluster->_pos.y()) > (_grid->getSizeY()/2.0) )
        correction.setY(_grid->getSizeY());
    return correction;

//    QVector3D distance(cluster->_pos-_pos);
//    _grid->correctDistance(distance);
//    return distance-(cluster->_pos-_pos);
}

QVector3D CellCluster::calcCellDistWithoutTorusCorrection (Cell* cell)
{
    return calcPosition(cell)-_pos;
}

QList< CellCluster* > CellCluster::decompose ()
{
    QList< CellCluster* > fragments;
    while( !_cells.isEmpty() ) {


        //find fragment
        QList< Cell* > component;
        quint64 tag(GlobalFunctions::getTag());
        getConnectedComponent(_cells[0], tag, component);

        //remove fragment from clusters
        QMap< quint64, CellCluster* > idClusterMap;
        foreach( Cell* cell, component) {
            idClusterMap[cell->getCluster()->_id] = cell->getCluster();
        }

        foreach( CellCluster* cluster, idClusterMap.values() ) {
            QMutableListIterator<Cell*> i(cluster->_cells);
            while (i.hasNext()) {
                if( i.next()->getTag() == tag )
                    i.remove();
            }
        }
        fragments << new CellCluster(component, _angle, _grid);
    }
    return fragments;
}

qreal CellCluster::calcAngularMassWithNewParticle (QVector3D particlePos)
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
    _grid->correctDisplacement(diff);
    qreal aMass = diff.lengthSquared();
    foreach(Cell* cell, _cells) {
        diff = cell->getRelPos() - center;
        _grid->correctDisplacement(diff);
        aMass = aMass + diff.lengthSquared();
    }
    return aMass;
}

qreal CellCluster::calcAngularMassWithoutUpdate ()
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
        _grid->correctDisplacement(displacement);
        aMass = aMass + displacement.lengthSquared();
    }
    return aMass;
}

bool CellCluster::isEmpty()
{
    return _cells.isEmpty();
}

QList< Cell* >& CellCluster::getCells ()
{
    return _cells;
}

/*QVector3D CellCluster::getCoordinate (Cell* cell)
{
    return _transform.map(cell->getRelPos());
}
*/

void CellCluster::findNearestCells (QVector3D pos, Cell*& cell1, Cell*& cell2)
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

const quint64& CellCluster::getId ()
{
    return _id;
}

void CellCluster::setId (quint64 id)
{
    _id = id;
}

QList< quint64 > CellCluster::getCellIds ()
{
    QList< quint64 > ids;
    foreach( Cell* cell, _cells )
        ids << cell->getId();
    return ids;
}

const quint64& CellCluster::getColor ()
{
    return _color;
}

QVector3D CellCluster::getPosition ()
{
    return _pos;
}

void CellCluster::setPosition (QVector3D pos, bool updateTransform)
{
    _pos = pos;
    if( updateTransform )
        calcTransform();
}

qreal CellCluster::getAngle ()
{
    return _angle;
}

void CellCluster::setAngle (qreal angle, bool updateTransform)
{
    _angle = angle;
    if( updateTransform )
        calcTransform();
}

QVector3D CellCluster::getVel ()
{
    return _vel;
}

void CellCluster::setVel (QVector3D vel)
{
    _vel = vel;
}

qreal CellCluster::getMass ()
{
    return _cells.size();
}

qreal CellCluster::getAngularVel ()
{
    return _angularVel;
}

void CellCluster::setAngularVel (qreal vel)
{
    _angularVel = vel;
}

qreal CellCluster::getAngularMass ()
{
    return _angularMass;
}


void CellCluster::calcTransform ()
{
    _transform.setToIdentity();
    _transform.translate(_pos);
    _transform.rotate(_angle, 0.0, 0.0, 1.0);
}

QVector3D CellCluster::relToAbsPos (QVector3D relPos)
{
    return _transform.map(relPos);
}

QVector3D CellCluster::absToRelPos (QVector3D absPos)
{
    return _transform.inverted().map(absPos);
}

void CellCluster::serialize (QDataStream& stream)
{
    stream << _angle << _pos << _angularVel << _vel;
    stream << _cells.size();
    ModelFacade *facade = ServiceLocator::getInstance().getService<ModelFacade>();
    foreach( Cell* cell, _cells ) {
        facade->serializeFeaturedCell(cell, stream);
    }
    stream << _id;
    stream << _color;
}


Cell* CellCluster::findNearestCell (QVector3D pos)
{
    foreach( Cell* cell, _cells)
        if((calcPosition(cell, true)-pos).lengthSquared() < 0.5)
            return cell;
    return 0;
}

void CellCluster::getConnectedComponent(Cell* cell, QList< Cell* >& component)
{
    component.clear();
    quint64 tag(GlobalFunctions::getTag());
    getConnectedComponent(cell, tag, component);
}

void CellCluster::getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component)
{
    if( cell->getTag() != tag ) {
        cell->setTag(tag);
        component << cell;
        for( int i = 0; i < cell->getNumConnections(); ++i ) {
            getConnectedComponent(cell->getConnection(i), tag, component);
        }
    }
}

CellCluster::CellCluster (Grid*& grid)
    : _grid(grid),
      _angle(0.0),
      _pos(0.0, 0.0, 0.0),
      _angularVel(0.0),
      _vel(0.0, 0.0, 0.0),
      _angularMass(0.0),
      _id(GlobalFunctions::getTag()),
      _color(GlobalFunctions::getTag())
{
    calcTransform();
}

CellCluster::CellCluster(QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel, Grid*& grid)
    : _grid(grid),
      _angle(angle),
      _pos(pos),
      _angularVel(angularVel),
      _vel(vel),
      _cells(cells),
      _id(GlobalFunctions::getTag()),
      _color(GlobalFunctions::getTag())
{
    grid->correctPosition(_pos);
    foreach(Cell* cell, _cells) {
        cell->setCluster(this);
    }
    calcTransform();
    updateRelCoordinates();
    updateAngularMass();
}

CellCluster::CellCluster (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid*& grid)
    : _grid(grid)
{
    //read data and reconstructing structures
    QMap< quint64, QList< quint64 > > connectingCells;
    QMap< quint64, Cell* > idCellMap;
    qreal angle(0);
    QVector3D pos;
    stream >> angle >> pos >> _angularVel >> _vel;
    setPosition(pos);
    setAngle(angle);
    int numCells(0);
    stream >> numCells;
    ModelFacade* facade = ServiceLocator::getInstance().getService<ModelFacade>();
    for(int i = 0; i < numCells; ++i ) {
        Cell* cell = facade->buildFeaturedCell(stream, connectingCells, _grid);
        cell->setCluster(this);
        _cells << cell;
        idCellMap[cell->getId()] = cell;

        //assigning new cell id
        quint64 newId = GlobalFunctions::getTag();
        oldNewCellIdMap[cell->getId()] = newId;
        oldIdCellMap[cell->getId()] = cell;
        cell->setId(newId);
    }
    quint64 oldClusterId(0);
    stream >> oldClusterId;
    stream >> _color;

    //assigning new cluster id
    _id = GlobalFunctions::getTag();
    oldNewClusterIdMap[oldClusterId] = _id;

    QMapIterator< quint64, QList< quint64 > > it(connectingCells);
    while (it.hasNext()) {
        it.next();
        Cell* cell(idCellMap[it.key()]);
        QList< quint64 > cellIdList(it.value());
        int i(0);
        foreach(quint64 cellId, cellIdList) {
            cell->setConnection(i, idCellMap[cellId]);
            ++i;
        }
    }

    updateRelCoordinates();
    updateAngularMass();
}

CellCluster::CellCluster(QList< Cell* > cells, qreal angle, Grid*& grid)
    : _grid(grid),
      _angle(angle),
      _cells(cells),
      _id(GlobalFunctions::getTag()),
      _color(GlobalFunctions::getTag())
{
    //calc new center
    QVector3D center(0.0,0.0,0.0);
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

void CellCluster::radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle)
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
    if( (qreal)qrand()/RAND_MAX < radFrequency) {
        radEnergy = radEnergy / radFrequency;
        radEnergy = radEnergy *2.0 * ((qreal)qrand()/RAND_MAX);
        if( radEnergy > (energy-1.0) )
            radEnergy = energy-1.0;
        energy = energy - radEnergy;

        //create energy particle with radEnergy
        QVector3D velPerturbation(((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB,
                                  ((qreal)qrand()/RAND_MAX-0.5)*simulationParameters.CELL_RAD_ENERGY_VEL_PERTURB, 0.0);
        QVector3D posPerturbation = velPerturbation.normalized();
        energyParticle = new EnergyParticle(radEnergy, calcPosition(originCell, _grid)+posPerturbation
            , originCell->getVel()*simulationParameters.CELL_RAD_ENERGY_VEL_MULT+velPerturbation, _grid);
        energyParticle->color = originCell->getColor();
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
                                             calcPosition(originCell, grid)+posPerturbation,
                                             originCell->_vel*simulationParameters.CELL_RAD_ENERGY_VEL_MULT+velPerturbation);
            energyParticle->color = originCell->getColor();
//        }
    }
*/
}





/*
    //apply gravitational forces
    QVector3D gSource1(200.0+qSin(0.5*degToRad*(qreal)time)*50, 200.0+qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    QVector3D gSource2(200.0-qSin(0.5*degToRad*(qreal)time)*50, 200.0-qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    foreach( Cell* cell, _cells) {
        QVector3D distance1 = gSource1-calcPosition(cell);
        QVector3D distance2 = gSource1-(calcPosition(cell)+cell->_vel);
        grid->correctDistance(distance1);
        grid->correctDistance(distance2);
        cell->_vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
        cell->_vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
        distance1 = gSource2-calcPosition(cell);
        distance2 = gSource2-(calcPosition(cell)+cell->_vel);
        grid->correctDistance(distance1);
        grid->correctDistance(distance2);
        cell->_vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
        cell->_vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
    }

    //update velocity and angular velocity
    updateVel_angularVel_via_cellVelocities();
*/


