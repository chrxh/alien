#include "shapeuniverse.h"

#include "aliencellgraphicsitem.h"
#include "aliencellconnectiongraphicsitem.h"
#include "alienenergygraphicsitem.h"
#include "markergraphicsitem.h"

#include "../../globaldata/editorsettings.h"
#include "../../globaldata/metadatamanager.h"
#include "../../globaldata/simulationparameters.h"

#include "../../simulation/entities/aliencellcluster.h"
#include "../../simulation/entities/alienenergy.h"
#include "../../simulation/entities/aliengrid.h"

#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>

ShapeUniverse::ShapeUniverse(QObject *parent) :
    QGraphicsScene(parent), _grid(0), _marker(0), _focusCenterCellItem(0)
{
    setBackgroundBrush(QBrush(QColor(0,0,0x30)));
}


void ShapeUniverse::universeUpdated (AlienGrid* grid)
{
    _grid = grid;
    if( !_grid )
        return;

    grid->lockData();
    _focusCells.clear();
    _focusEnergyParticles.clear();
    _highlightedCells.clear();
    _highlightedEnergyParticles.clear();
    _cellItems.clear();
    _connectionItems.clear();
    if( _marker) {
        delete _marker;
        _marker = 0;
    }

    AlienCell* focusCenterCell(0);
    if( _focusCenterCellItem)
        focusCenterCell = _focusCenterCellItem->getCell();
    _focusCenterCellItem = 0;

    //reset scene
    clear();
    setSceneRect(0,0,grid->getSizeX(),grid->getSizeY());

    //draw boundaries
    QGraphicsScene::addRect(0.0, 0.0, grid->getSizeX(), grid->getSizeY(), QPen(QColor(0, 0, 0x80)));

    //draw energy particles
    foreach( AlienEnergy* energy, grid->getEnergyParticles() ) {
        createEnergyItem(energy);
    }

    //draw cell clusters
    foreach( AlienCellCluster* cluster, grid->getClusters() ) {
        foreach( AlienCell* cell, cluster->getCells()) {

            //create connections between cells
            for(int i = 0; i < cell->getNumConnections(); ++i ) {
                AlienCell* otherCell(cell->getConnection(i));

                //otherCell not already drawn?
                if( !_cellItems.contains(otherCell->getId()) ) {
                    createConnectionItem(cell, otherCell);
                }
            }

            //create graphic representation of cell
            AlienCellGraphicsItem* cellItem = createCellItem(cell);

            //remember the cell item which should be focused
            if( cell == focusCenterCell )
                _focusCenterCellItem = cellItem;
        }
    }

    //set cell color according to the meta data
    setCellColorFromMetadata();

    grid->unlockData();
    update();
}

void ShapeUniverse::cellCreated (AlienCell* cell)
{
    if( (!_grid) || (!cell) )
        return;

    _grid->lockData();
    createCellItem(cell);

    //remember focus cell
    _focusCells.clear();
    _focusEnergyParticles.clear();
    _focusCells << _cellItems[cell->getId()];

    //highlight cell
    unhighlight();
    highlightCell(cell);

    _grid->unlockData();
    QGraphicsScene::update();
}

void ShapeUniverse::energyParticleCreated (AlienEnergy* e)
{
    if( (!_grid) || (!e) )
        return;

    _grid->lockData();
    _focusCells.clear();
    _focusEnergyParticles.clear();

    //create graphic item
    AlienEnergyGraphicsItem* eItem = createEnergyItem(e);
    _focusEnergyParticles << eItem;

    //highlight energy particle
    unhighlight();
    highlightEnergyParticle(eItem);

    _grid->unlockData();
    QGraphicsScene::update();
}

void ShapeUniverse::defocused ()
{
    //remove hightlighting
    unhighlight();
    _focusCells.clear();
    _focusEnergyParticles.clear();
    QGraphicsScene::update();
}

void ShapeUniverse::energyParticleUpdated_Slot (AlienEnergy* e)
{
    if( !_grid )
        return;
    _grid->lockData();

    if( _energyItems.contains(e->id) ) {
        QVector3D pos = e->pos;
        AlienEnergyGraphicsItem* eItem = _energyItems[e->id];
        _grid->correctPosition(pos);
        eItem->setPos(pos.x(), pos.y());
    }
    _grid->unlockData();

    QGraphicsScene::update();
}

void ShapeUniverse::getExtendedSelection (QList< AlienCellCluster* >& clusters, QList< AlienEnergy* >& es)
{
    //extract selected cluster
    _grid->lockData();
    QMap< quint64, AlienCellCluster* > idClusterMap;
    QList< AlienCellGraphicsItem* > highlightedCells = _highlightedCells.values();
    foreach( AlienCellGraphicsItem* cellItem, highlightedCells ) {
        AlienCellCluster* cluster = cellItem->getCell()->getCluster();
        idClusterMap[cluster->getId()] = cluster;
    }
    _grid->unlockData();
    clusters = idClusterMap.values();

    //selected energy particles
    QList< AlienEnergyGraphicsItem* > highlightedEs = _highlightedEnergyParticles.values();
    foreach (AlienEnergyGraphicsItem* eItem, highlightedEs) {
        es << eItem->getEnergyParticle();
    }
}

void ShapeUniverse::delSelection (QList< AlienCell* >& cells, QList< AlienEnergy* >& es)
{
    _grid->lockData();

    //remove highlighting (has to be done first since the cells will be deleted in the following!!!)
    unhighlight();

    //del focused cells with connections
    foreach( AlienCellGraphicsItem* cellItem, _focusCells ) {
        cells << cellItem->getCell();
        quint64 cellId = cellItem->getCell()->getId();

        //del cell connections
        delConnectionItem(cellId);

        //del cell
        _cellItems.remove(cellId);
        delete cellItem;
    }
    _focusCells.clear();

    //del focused energy particles
    foreach( AlienEnergyGraphicsItem* eItem, _focusEnergyParticles ) {
        es << eItem->getEnergyParticle();
        _energyItems.remove(eItem->getEnergyParticle()->id);
        delete eItem;
    }
    _focusEnergyParticles.clear();
    _grid->unlockData();
    QGraphicsScene::update();
}

void ShapeUniverse::delExtendedSelection (QList< AlienCellCluster* >& clusters, QList< AlienEnergy* >& es)
{
    _grid->lockData();

    //identify all cells and their clusters which should be deleted
    QSet< quint64 > cellsToBeDeleted;
    QSet< quint64 > clustersToBeDeleted;
    QMap< quint64, AlienCellCluster* > idClusterMap;
    foreach(AlienCellGraphicsItem* cellItem, _focusCells) {
        AlienCellCluster* cluster = cellItem->getCell()->getCluster();
        clustersToBeDeleted << cluster->getId();
        idClusterMap[cluster->getId()] = cluster;
        foreach( AlienCell* cell, cluster->getCells()) {
            cellsToBeDeleted << cell->getId();
        }
    }
    _focusCells.clear();
    _highlightedCells.clear();
    foreach(quint64 clusterId, clustersToBeDeleted)
        clusters << idClusterMap[clusterId];

    //delete graphic cells and their connections
    foreach(quint64 cellId, cellsToBeDeleted) {

        //del cell
        AlienCellGraphicsItem* cellItem = _cellItems.take(cellId);
        if( cellItem )
            delete cellItem;

        //del cell connections
        QMap< quint64, AlienCellConnectionGraphicsItem* > items = _connectionItems.take(cellId);
        if( !items.empty() ) {
            foreach(AlienCellConnectionGraphicsItem* conItem, items.values()) {
                delete conItem;
            }
            foreach(quint64 key, items.keys()) {
                _connectionItems[key].remove(cellId);
                if( _connectionItems[key].empty() )
                    _connectionItems.remove(key);
            }
        }
    }

    //del focused energy particles
    foreach( AlienEnergyGraphicsItem* eItem, _focusEnergyParticles ) {
        es << eItem->getEnergyParticle();
        _energyItems.remove(eItem->getEnergyParticle()->id);
        delete eItem;
    }
    _focusEnergyParticles.clear();
    _highlightedEnergyParticles.clear();
    _grid->unlockData();
    QGraphicsScene::update();
}

void ShapeUniverse::metadataUpdated ()
{
    //set cell colors
    _grid->lockData();
    setCellColorFromMetadata();
    _grid->unlockData();

    QGraphicsScene::update();
}

QGraphicsItem* ShapeUniverse::getFocusCenterCell ()
{
    return _focusCenterCellItem;
}

void ShapeUniverse::reclustered (QList< AlienCellCluster* > clusters)
{
    if( !_grid )
        return;

    _grid->lockData();

     //remove hightlighting
    unhighlight();

    //move graphic cells corresponding to the AlienCells in "clusters" and delete their connections
    foreach(AlienCellCluster* cluster, clusters) {
        foreach(AlienCell* cell, cluster->getCells()) {

            //move cell
            if( _cellItems.contains(cell->getId()) ) {
                QVector3D pos = cell->calcPosition();
                AlienCellGraphicsItem* cellItem = _cellItems[cell->getId()];
                _grid->correctPosition(pos);
                cellItem->setPos(pos.x(), pos.y());
                cellItem->setNumToken(cell->getNumToken());
                bool connectable = (cell->getNumConnections() < cell->getMaxConnections());
                cellItem->setConnectable(connectable);
            }

            //not available? => create
            else {
                createCellItem(cell);
            }

            //del cell connections
            QMap< quint64, AlienCellConnectionGraphicsItem* > items = _connectionItems.take(cell->getId());
            if( !items.empty() ) {
                foreach(AlienCellConnectionGraphicsItem* conItem, items.values()) {
                    delete conItem;
                }
                foreach(quint64 key, items.keys()) {
                    _connectionItems[key].remove(cell->getId());
                    if( _connectionItems[key].empty() )
                        _connectionItems.remove(key);
                }
            }
        }
    }

    //draw cell connection
    foreach( AlienCellCluster* cluster, clusters ) {
        foreach( AlienCell* cell, cluster->getCells()) {

            //create connections between cells
            for(int i = 0; i < cell->getNumConnections(); ++i ) {
                AlienCell* otherCell(cell->getConnection(i));

                //otherCell not already drawn?
                if( !_connectionItems[cell->getId()].contains(otherCell->getId()) ) {
                    createConnectionItem(cell, otherCell);
                }
            }

        }
    }

    //highlight cells, clusters and energy particles
//    unhighlight();
    foreach(AlienCellGraphicsItem* cellItem, _focusCells)
        highlightCell(cellItem->getCell());
    foreach(AlienEnergyGraphicsItem* eItem, _focusEnergyParticles)
        highlightEnergyParticle(eItem);

    _grid->unlockData();
    QGraphicsScene::update();
}


void ShapeUniverse::mousePressEvent (QGraphicsSceneMouseEvent* e)
{
    if( !_grid )
        return;
    _grid->lockData();

    bool _clickedOnSomething = false;
    QList< QGraphicsItem* > items(QGraphicsScene::items(e->scenePos()));
    foreach(QGraphicsItem* item, items ) {

        //clicked on cell item?
        AlienCellGraphicsItem* cellItem = qgraphicsitem_cast<AlienCellGraphicsItem*>(item);
        if( cellItem ) {
            _clickedOnSomething = true;

            //focus new single cell if it is not already focused
            if( !_focusCells.contains(cellItem) ) {
                unhighlight();
                highlightCell(cellItem->getCell());
                _focusCells.clear();
                _focusEnergyParticles.clear();
                _focusCells << cellItem;
                QGraphicsScene::update();
                emit focusCell(cellItem->getCell());
                _focusCenterCellItem = cellItem;
            }
            break;
        }

        //clicked on energy particle item?
        AlienEnergyGraphicsItem* eItem = qgraphicsitem_cast<AlienEnergyGraphicsItem*>(item);
        if( eItem ) {
            _clickedOnSomething = true;

            //focus new single particle if it is not already focused
            if( !_focusEnergyParticles.contains(eItem) ) {
                unhighlight();
                highlightEnergyParticle(eItem);
                _focusCells.clear();
                _focusEnergyParticles.clear();
                _focusEnergyParticles << eItem;
                QGraphicsScene::update();
                emit focusEnergyParticle(eItem->getEnergyParticle());
            }
            break;
        }
    }

/*    if( temp == 1 ) {
        qDebug("h: %d", _highlightedCells.size());
        unhighlight();
        QGraphicsScene::update();
        _grid->unlockData();
        return;
    }
*/
    //nothing clicked? => defocus
    if( !_clickedOnSomething ) {
        unhighlight();
        _focusCells.clear();
        _focusEnergyParticles.clear();
        emit defocus();
        _focusCenterCellItem = 0;

        //create marker
        if( _marker )
            delete _marker;
        qreal x = e->scenePos().x();
        qreal y = e->scenePos().y();
        _marker = new MarkerGraphicsItem(x, y, x, y);
        QGraphicsScene::addItem(_marker);
        QGraphicsScene::update();
    }

    _grid->unlockData();
}

void ShapeUniverse::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
    if( _marker ) {
        delete _marker;
        _marker = 0;

        if( _focusCells.isEmpty() && _focusEnergyParticles.isEmpty() ) {
            emit defocus();
            _focusCenterCellItem = 0;
        }
    }
}

void ShapeUniverse::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    if( !_grid )
        return;

//    _grid->lockData();

    //mouse buttons
    bool leftButton = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
    bool rightButton = ((e->buttons() & Qt::RightButton) == Qt::RightButton);

    if( leftButton || rightButton ) {

        //move marker?
        if( _marker ) {
            _grid->lockData();

            //set pos
            _marker->setEndPos(e->scenePos().x(), e->scenePos().y());

            //unhighlight cells/particles
            unhighlight();
            _focusCells.clear();
            _focusEnergyParticles.clear();

            //obtain colliding items
            QList< QGraphicsItem* > items = _marker->collidingItems();
            foreach( QGraphicsItem* item, items ){

                //cell item?
                AlienCellGraphicsItem* cellItem = qgraphicsitem_cast<AlienCellGraphicsItem*>(item);
                if( cellItem ) {

                    //highlight cell
                    highlightCell(cellItem->getCell());
                    _focusCells << cellItem;
                }

                //energy item?
                AlienEnergyGraphicsItem* eItem = qgraphicsitem_cast<AlienEnergyGraphicsItem*>(item);
                if( eItem ) {

                    //highlight new particle
                    highlightEnergyParticle(eItem);
                    _focusEnergyParticles << eItem;
                }
            }

            _grid->unlockData();
            emit entitiesSelected(_focusCells.size(), _focusEnergyParticles.size());
            QGraphicsScene::update();
        }
        else {

            QPointF lastPos = e->lastScenePos();
            QPointF pos = e->scenePos();
            QVector3D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y(), 0.0);

            //calc rotation matrix (used when both mouse buttons are pressed)
            QVector3D center = calcCenterOfHighlightedObjects();
            qreal angleDelta = delta.y()*20.0;
            QMatrix4x4 transform;
            transform.setToIdentity();
            transform.translate(center);
            transform.rotate(angleDelta, 0.0, 0.0, 1.0);
            transform.translate(-center);


            //update focused energy particles
            foreach( AlienEnergyGraphicsItem* eItem, _focusEnergyParticles ) {

                //update new position to the energy particle on our own
                AlienEnergy* energy = eItem->getEnergyParticle();
                _grid->lockData();
                _grid->setEnergy(energy->pos, 0);

                //not [left and right] mouse button pressed?
                if( (!leftButton) || (!rightButton) ) {
                    energy->pos = energy->pos + delta;
                }
                else {
                    energy->pos = transform.map(energy->pos);
                }

                _grid->setEnergy(energy->pos, energy);
                _grid->unlockData();
//                QPointF p = eItem->pos();
                eItem->setPos(energy->pos.x(), energy->pos.y());

                //inform other instances about cell cluster changes
                emit energyParticleUpdated(energy);
            }

            QList< AlienCell* > cells;
            QList< AlienCellReduced > newCellsData;

            //update focused cells
            foreach( AlienCellGraphicsItem* cellItem, _focusCells ) {

                //retrieve cell information
                AlienCell* cell = cellItem->getCell();
                _grid->lockData();
                AlienCellReduced newCellData(cell);
                _grid->unlockData();

                //only left mouse button pressed?
                if( leftButton && (!rightButton) ) {

                    //update new position to the cell
                    newCellData.cellPos += delta;

                    //invoke reclustering
                    cells << cell;
                    newCellsData << newCellData;
                }

                //only right mouse button pressed?
                if( (!leftButton) && rightButton ) {

                    //update new position to the cell cluster
                    newCellData.clusterPos += delta;

                    //inform other instances about cell cluster changes
                    cells << cell;
                    newCellsData << newCellData;
                }

                //left and right mouse button pressed?
                if( leftButton && rightButton ) {

                    //1. rotate around own cluster center
                    newCellData.clusterAngle += angleDelta;

                    //2. rotate around common center
                    newCellData.clusterPos = transform.map(newCellData.clusterPos);

                    //inform other instances about cell cluster changes
                    cells << cell;
                    newCellsData << newCellData;
                }
            }

            //sending changes to simulator
            if( !cells.isEmpty() ) {
                if( leftButton && (!rightButton) )
                    emit updateCell(cells, newCellsData, false);
                if( (!leftButton) && rightButton )
                    emit updateCell(cells, newCellsData, true);
                if( leftButton && rightButton )
                    emit updateCell(cells, newCellsData, true);
            }
        }
    }
//    _grid->unlockData();
}

AlienEnergyGraphicsItem* ShapeUniverse::createEnergyItem (AlienEnergy* e)
{
    //create item
    QVector3D pos(e->pos);
    AlienEnergyGraphicsItem* eItem = new AlienEnergyGraphicsItem(e, pos.x(), pos.y());
    QGraphicsScene::addItem(eItem);

    //register item
    _energyItems[e->id] = eItem;
    return eItem;
}

AlienCellGraphicsItem* ShapeUniverse::createCellItem (AlienCell* cell)
{
    //create item
    QVector3D pos(cell->calcPosition());
    bool connectable = (cell->getNumConnections() < cell->getMaxConnections());
    AlienCellGraphicsItem* cellItem = new AlienCellGraphicsItem(cell, pos.x(), pos.y(), connectable, cell->getNumToken(), cell->getColor());
    QGraphicsScene::addItem(cellItem);

    //register item
    _cellItems[cell->getId()] = cellItem;
    return cellItem;
}

void ShapeUniverse::createConnectionItem (AlienCell* cell, AlienCell* otherCell)
{
    QVector3D pos(cell->getCluster()->calcPosition(cell));
    QVector3D otherPos(otherCell->getCluster()->calcPosition(otherCell));

    //directed connection?
    AlienCellConnectionGraphicsItem::ConnectionState s = AlienCellConnectionGraphicsItem::NO_DIR_CONNECTION;
    if( cell->getTokenAccessNumber() == ((otherCell->getTokenAccessNumber()+1)%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS) && (!cell->blockToken()) ) {
        s = AlienCellConnectionGraphicsItem::B_TO_A_CONNECTION;
    }
    if( ((cell->getTokenAccessNumber()+1)%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber() && (!otherCell->blockToken()) ) {
        s = AlienCellConnectionGraphicsItem::A_TO_B_CONNECTION;
    }
    AlienCellConnectionGraphicsItem* connectionItem = new AlienCellConnectionGraphicsItem(pos.x(), pos.y(), otherPos.x(), otherPos.y(), s);
    QGraphicsScene::addItem(connectionItem);

    //register connection
    _connectionItems[cell->getId()][otherCell->getId()] = connectionItem;
    _connectionItems[otherCell->getId()][cell->getId()] = connectionItem;
}

void ShapeUniverse::delConnectionItem (quint64 cellId)
{
    QMap< quint64, AlienCellConnectionGraphicsItem* > items = _connectionItems.take(cellId);
    if( !items.empty() ) {
        foreach(AlienCellConnectionGraphicsItem* conItem, items.values()) {
            delete conItem;
        }
        foreach(quint64 key, items.keys()) {
            _connectionItems[key].remove(cellId);
            if( _connectionItems[key].empty() )
                _connectionItems.remove(key);
        }
    }
}

void ShapeUniverse::unhighlight ()
{
    //defocus old cells
    QList< AlienCellGraphicsItem* > highlightedCells = _highlightedCells.values();
    foreach(AlienCellGraphicsItem* cellItem, highlightedCells) {
        cellItem->setFocusState(AlienCellGraphicsItem::NO_FOCUS);
    }
    _highlightedCells.clear();

    //defocus old energy particles
    QList< AlienEnergyGraphicsItem* > highlightedEs = _highlightedEnergyParticles.values();
    foreach(AlienEnergyGraphicsItem* eItem, highlightedEs) {
        eItem->setFocusState(AlienEnergyGraphicsItem::NO_FOCUS);
    }
    _highlightedEnergyParticles.clear();
}

void ShapeUniverse::highlightCell (AlienCell* cell)
{
    if( !cell )
        return;

    //focus cellcluster
    foreach(AlienCell* otherCell, cell->getCluster()->getCells()) {
        if( _cellItems.contains(otherCell->getId()) ) {
            AlienCellGraphicsItem* cellItem = _cellItems[otherCell->getId()];
            if( cellItem->getFocusState() == AlienCellGraphicsItem::NO_FOCUS )
                cellItem->setFocusState(AlienCellGraphicsItem::FOCUS_CLUSTER);
            _highlightedCells[otherCell->getId()] = cellItem;
        }
    }

    //focus cell
    if( _cellItems.contains(cell->getId()) )
        _cellItems[cell->getId()]->setFocusState(AlienCellGraphicsItem::FOCUS_CELL);
}

void ShapeUniverse::highlightEnergyParticle (AlienEnergyGraphicsItem* e)
{
    if( !e )
        return;

    //focus energy particle
    e->setFocusState(AlienEnergyGraphicsItem::FOCUS);
    _highlightedEnergyParticles[e->getEnergyParticle()->id] = e;
}

void ShapeUniverse::setCellColorFromMetadata ()
{
    //set cell colors
    QMapIterator< quint64, AlienCellGraphicsItem* > it(_cellItems);
    while( it.hasNext() ) {
        it.next();
        AlienCellGraphicsItem* cellItem = it.value();
        cellItem->setColor(cellItem->getCell()->getColor());
    }
}

QVector3D ShapeUniverse::calcCenterOfHighlightedObjects ()
{
    QVector3D center;
    QList< AlienCellGraphicsItem* > cellItems(_highlightedCells.values());
    foreach( AlienCellGraphicsItem* cellItem, cellItems )
        center += QVector3D(cellItem->pos().x(), cellItem->pos().y(), 0.0);
    QList< AlienEnergyGraphicsItem* > eItems(_highlightedEnergyParticles.values());
    foreach( AlienEnergyGraphicsItem* eItem, eItems )
        center += QVector3D(eItem->pos().x(), eItem->pos().y(), 0.0);
    return center/(cellItems.size()+eItems.size());
}


