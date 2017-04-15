#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "global/servicelocator.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/features/cellfunction.h"
#include "model/simulationparameters.h"
#include "model/config.h"
#include "model/alienfacade.h"
#include "model/simulationcontext.h"
#include "model/topology.h"
#include "model/energyparticlemap.h"
#include "gui/guisettings.h"
#include "gui/guisettings.h"

#include "cellgraphicsitem.h"
#include "cellconnectiongraphicsitem.h"
#include "cellgraphicsitemconfig.h"
#include "energygraphicsitem.h"
#include "markergraphicsitem.h"
#include "shapeuniverse.h"

ShapeUniverse::ShapeUniverse(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(QColor(0,0,0x30)));
	_itemConfig = new CellGraphicsItemConfig();
}

ShapeUniverse::~ShapeUniverse()
{
	delete _itemConfig;
}


void ShapeUniverse::universeUpdated (SimulationContext* context)
{
    _context = context;
    if( !_context)
        return;

	_context->lock();
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

    Cell* focusCenterCell(0);
    if( _focusCenterCellItem)
        focusCenterCell = _focusCenterCellItem->getCell();
    _focusCenterCellItem = 0;

    //reset scene
    clear();
	IntVector2D size = _context->getTopology()->getSize();
    setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);

    //draw boundaries
    QGraphicsScene::addRect(0.0, 0.0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE, QPen(QColor(0, 0, 0x80)));

    //draw energy particles
    foreach( EnergyParticle* energy, _context->getEnergyParticlesRef() ) {
        createEnergyItem(energy);
    }

    //draw cell clusters
    foreach( CellCluster* cluster, _context->getClustersRef() ) {
        foreach( Cell* cell, cluster->getCellsRef()) {

            //create connections between cells
            for(int i = 0; i < cell->getNumConnections(); ++i ) {
                Cell* otherCell(cell->getConnection(i));

                //otherCell not already drawn?
                if( !_cellItems.contains(otherCell->getId()) ) {
                    createConnectionItem(cell, otherCell);
                }
            }

            //create graphic representation of cell
            CellGraphicsItem* cellItem = createCellItem(cell);

            //remember the cell item which should be focused
            if( cell == focusCenterCell )
                _focusCenterCellItem = cellItem;
        }
    }

    //set cell color according to the meta data
    setCellColorFromMetadata();

    _context->unlock();
    update();
}

void ShapeUniverse::cellCreated (Cell* cell)
{
    if( (!_context) || (!cell) )
        return;

	_context->lock();
    createCellItem(cell);

    //remember focus cell
    _focusCells.clear();
    _focusEnergyParticles.clear();
    _focusCells << _cellItems[cell->getId()];

    //highlight cell
    unhighlight();
    highlightCell(cell);

	_context->unlock();
    QGraphicsScene::update();
}

void ShapeUniverse::energyParticleCreated (EnergyParticle* e)
{
    if( (!_context) || (!e) )
        return;

	_context->lock();
    _focusCells.clear();
    _focusEnergyParticles.clear();

    //create graphic item
    EnergyGraphicsItem* eItem = createEnergyItem(e);
    _focusEnergyParticles << eItem;

    //highlight energy particle
    unhighlight();
    highlightEnergyParticle(eItem);

	_context->unlock();
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

void ShapeUniverse::energyParticleUpdated_Slot (EnergyParticle* e)
{
    if( !_context)
        return;
	_context->lock();

    if( _energyItems.contains(e->getId()) ) {
        QVector3D pos = e->getPosition();
        EnergyGraphicsItem* eItem = _energyItems[e->getId()];
		_context->getTopology()->correctPosition(pos);
        eItem->setPos(pos.x()*GRAPHICS_ITEM_SIZE, pos.y()*GRAPHICS_ITEM_SIZE);
    }
	_context->unlock();

    QGraphicsScene::update();
}

void ShapeUniverse::getExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es)
{
    //extract selected cluster
	_context->lock();
    QMap< quint64, CellCluster* > idClusterMap;
    QList< CellGraphicsItem* > highlightedCells = _highlightedCells.values();
    foreach( CellGraphicsItem* cellItem, highlightedCells ) {
        CellCluster* cluster = cellItem->getCell()->getCluster();
        idClusterMap[cluster->getId()] = cluster;
    }
	_context->unlock();
    clusters = idClusterMap.values();

    //selected energy particles
    QList< EnergyGraphicsItem* > highlightedEs = _highlightedEnergyParticles.values();
    foreach (EnergyGraphicsItem* eItem, highlightedEs) {
        es << eItem->getEnergyParticle();
    }
}

void ShapeUniverse::delSelection (QList< Cell* >& cells, QList< EnergyParticle* >& es)
{
	_context->lock();

    //remove highlighting (has to be done first since the cells will be deleted in the following!!!)
    unhighlight();

    //del focused cells with connections
    foreach( CellGraphicsItem* cellItem, _focusCells ) {
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
    foreach( EnergyGraphicsItem* eItem, _focusEnergyParticles ) {
        es << eItem->getEnergyParticle();
        _energyItems.remove(eItem->getEnergyParticle()->getId());
        delete eItem;
    }
    _focusEnergyParticles.clear();
	_context->unlock();
    QGraphicsScene::update();
}

void ShapeUniverse::delExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es)
{
    _context->lock();

    //identify all cells and their clusters which should be deleted
    std::set<quint64> cellsToBeDeleted;
    std::set<quint64> clustersToBeDeleted;
    QMap< quint64, CellCluster* > idClusterMap;
    foreach(CellGraphicsItem* cellItem, _focusCells) {
        CellCluster* cluster = cellItem->getCell()->getCluster();
        clustersToBeDeleted.insert(cluster->getId());
        idClusterMap[cluster->getId()] = cluster;
        foreach( Cell* cell, cluster->getCellsRef()) {
            cellsToBeDeleted.insert(cell->getId());
        }
    }
    _focusCells.clear();
    _highlightedCells.clear();
    foreach(quint64 clusterId, clustersToBeDeleted)
        clusters << idClusterMap[clusterId];

    //delete graphic cells and their connections
    foreach(quint64 cellId, cellsToBeDeleted) {

        //del cell
        CellGraphicsItem* cellItem = _cellItems.take(cellId);
        if( cellItem )
            delete cellItem;

        //del cell connections
        QMap< quint64, CellConnectionGraphicsItem* > items = _connectionItems.take(cellId);
        if( !items.empty() ) {
            foreach(CellConnectionGraphicsItem* conItem, items.values()) {
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
    foreach( EnergyGraphicsItem* eItem, _focusEnergyParticles ) {
        es << eItem->getEnergyParticle();
        _energyItems.remove(eItem->getEnergyParticle()->getId());
        delete eItem;
    }
    _focusEnergyParticles.clear();
    _highlightedEnergyParticles.clear();
    _context->unlock();
    QGraphicsScene::update();
}

void ShapeUniverse::metadataUpdated ()
{
    //set cell colors
    _context->lock();
    setCellColorFromMetadata();
    _context->unlock();

    QGraphicsScene::update();
}

QGraphicsItem* ShapeUniverse::getFocusCenterCell ()
{
    return _focusCenterCellItem;
}

void ShapeUniverse::toggleInformation(bool on)
{
	_itemConfig->showInfo = on;
	QGraphicsScene::update();
}

void ShapeUniverse::reclustered (QList< CellCluster* > clusters)
{
    if( !_context)
        return;

    _context->lock();

     //remove hightlighting
    unhighlight();

    //move graphic cells corresponding to the Cells in "clusters" and delete their connections
	Topology* topo = _context->getTopology();
    foreach(CellCluster* cluster, clusters) {
        foreach(Cell* cell, cluster->getCellsRef()) {

            //move cell
            if( _cellItems.contains(cell->getId()) ) {
                QVector3D pos = cell->calcPosition();
                CellGraphicsItem* cellItem = _cellItems[cell->getId()];
                topo->correctPosition(pos);
                cellItem->setPos(pos.x()*GRAPHICS_ITEM_SIZE, pos.y()*GRAPHICS_ITEM_SIZE);
                cellItem->setNumToken(cell->getNumToken());
				cellItem->setDisplayString(getCellFunctionString(cell));
				cellItem->setBranchNumber(cell->getBranchNumber());
                bool connectable = (cell->getNumConnections() < cell->getMaxConnections());
                cellItem->setConnectable(connectable);
            }

            //not available? => create
            else {
                createCellItem(cell);
            }

            //del cell connections
            QMap< quint64, CellConnectionGraphicsItem* > items = _connectionItems.take(cell->getId());
            if( !items.empty() ) {
                foreach(CellConnectionGraphicsItem* conItem, items.values()) {
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
    foreach( CellCluster* cluster, clusters ) {
        foreach( Cell* cell, cluster->getCellsRef()) {

            //create connections between cells
            for(int i = 0; i < cell->getNumConnections(); ++i ) {
                Cell* otherCell(cell->getConnection(i));

                //otherCell not already drawn?
                if( !_connectionItems[cell->getId()].contains(otherCell->getId()) ) {
                    createConnectionItem(cell, otherCell);
                }
            }

        }
    }

    //highlight cells, clusters and energy particles
//    unhighlight();
    foreach(CellGraphicsItem* cellItem, _focusCells)
        highlightCell(cellItem->getCell());
    foreach(EnergyGraphicsItem* eItem, _focusEnergyParticles)
        highlightEnergyParticle(eItem);

    _context->unlock();
    QGraphicsScene::update();
}


void ShapeUniverse::mousePressEvent (QGraphicsSceneMouseEvent* e)
{
    if( !_context )
        return;
    _context->lock();

    bool _clickedOnSomething = false;
    QList< QGraphicsItem* > items(QGraphicsScene::items(e->scenePos()));
    foreach(QGraphicsItem* item, items ) {

        //clicked on cell item?
        CellGraphicsItem* cellItem = qgraphicsitem_cast<CellGraphicsItem*>(item);
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
        EnergyGraphicsItem* eItem = qgraphicsitem_cast<EnergyGraphicsItem*>(item);
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
        _context->unlock();
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

    _context->unlock();
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
    if( !_context)
        return;

//    _context->lock();

    //mouse buttons
    bool leftButton = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
    bool rightButton = ((e->buttons() & Qt::RightButton) == Qt::RightButton);

    if( leftButton || rightButton ) {

        //move marker?
        if( _marker ) {
            _context->lock();

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
                CellGraphicsItem* cellItem = qgraphicsitem_cast<CellGraphicsItem*>(item);
                if( cellItem ) {

                    //highlight cell
                    highlightCell(cellItem->getCell());
                    _focusCells << cellItem;
                }

                //energy item?
                EnergyGraphicsItem* eItem = qgraphicsitem_cast<EnergyGraphicsItem*>(item);
                if( eItem ) {

                    //highlight new particle
                    highlightEnergyParticle(eItem);
                    _focusEnergyParticles << eItem;
                }
            }

            _context->unlock();
            emit entitiesSelected(_focusCells.size(), _focusEnergyParticles.size());
            QGraphicsScene::update();
        }
        else {

            QPointF lastPos = e->lastScenePos();
            QPointF pos = e->scenePos();
            QVector3D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y(), 0.0);
			delta = delta / GRAPHICS_ITEM_SIZE;

            //calc rotation matrix (used when both mouse buttons are pressed)
            QVector3D center = calcCenterOfHighlightedObjects();
            qreal angleDelta = delta.y()*20.0;
            QMatrix4x4 transform;
            transform.setToIdentity();
            transform.translate(center);
            transform.rotate(angleDelta, 0.0, 0.0, 1.0);
            transform.translate(-center);


            //update focused energy particles
			EnergyParticleMap* energyMap = _context->getEnergyParticleMap();
            foreach( EnergyGraphicsItem* eItem, _focusEnergyParticles ) {

                //update new position to the energy particle on our own
                EnergyParticle* energy = eItem->getEnergyParticle();
                _context->lock();
                energyMap->setParticle(energy->getPosition(), nullptr);

                //not [left and right] mouse button pressed?
                if( (!leftButton) || (!rightButton) ) {
                    energy->setPosition(energy->getPosition() + delta);
                }
                else {
                    energy->setPosition(transform.map(energy->getPosition()));
                }

                energyMap->setParticle(energy->getPosition(), energy);
                _context->unlock();
//                QPointF p = eItem->pos();
				auto particlePos = energy->getPosition();
                eItem->setPos(particlePos.x()*GRAPHICS_ITEM_SIZE, particlePos.y()*GRAPHICS_ITEM_SIZE);

                //inform other instances about cell cluster changes
                emit energyParticleUpdated(energy);
            }

            QList< Cell* > cells;
            QList< CellTO > newCellsData;

            //update focused cells
            AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
            foreach( CellGraphicsItem* cellItem, _focusCells ) {

                //retrieve cell information
                Cell* cell = cellItem->getCell();
                _context->lock();
                CellTO newCellData = facade->buildFeaturedCellTO(cell);
                _context->unlock();

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
//    _context->unlock();
}

EnergyGraphicsItem* ShapeUniverse::createEnergyItem (EnergyParticle* e)
{
    //create item
    QVector3D pos(e->getPosition());
    EnergyGraphicsItem* eItem = new EnergyGraphicsItem(e, pos.x()*GRAPHICS_ITEM_SIZE, pos.y()*GRAPHICS_ITEM_SIZE);
    QGraphicsScene::addItem(eItem);

    //register item
    _energyItems[e->getId()] = eItem;
    return eItem;
}

CellGraphicsItem* ShapeUniverse::createCellItem (Cell* cell)
{
    //create item
    QVector3D pos(cell->calcPosition());
    bool connectable = (cell->getNumConnections() < cell->getMaxConnections());
    CellGraphicsItem* cellItem = new CellGraphicsItem(_itemConfig, cell, pos.x() * GRAPHICS_ITEM_SIZE, pos.y() * GRAPHICS_ITEM_SIZE
		, connectable, cell->getNumToken(), cell->getMetadata().color, getCellFunctionString(cell), cell->getBranchNumber());
    QGraphicsScene::addItem(cellItem);

    //register item
    _cellItems[cell->getId()] = cellItem;
    return cellItem;
}

QString ShapeUniverse::getCellFunctionString(Cell * cell)
{
	CellFunction* cellFunction = cell->getFeatures()->findObject<CellFunction>();
	return Enums::getTypeString(cellFunction->getType());
}

void ShapeUniverse::createConnectionItem (Cell* cell, Cell* otherCell)
{
    QVector3D pos(cell->getCluster()->calcPosition(cell));
    QVector3D otherPos(otherCell->getCluster()->calcPosition(otherCell));

    //directed connection?
    CellConnectionGraphicsItem::ConnectionState s = CellConnectionGraphicsItem::NO_DIR_CONNECTION;
    if( cell->getBranchNumber() == ((otherCell->getBranchNumber()+1) % _context->getSimulationParameters()->MAX_TOKEN_ACCESS_NUMBERS) && (!cell->isTokenBlocked()) ) {
        s = CellConnectionGraphicsItem::B_TO_A_CONNECTION;
    }
    if( ((cell->getBranchNumber()+1) % _context->getSimulationParameters()->MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getBranchNumber() && (!otherCell->isTokenBlocked()) ) {
        s = CellConnectionGraphicsItem::A_TO_B_CONNECTION;
    }
    CellConnectionGraphicsItem* connectionItem = new CellConnectionGraphicsItem(pos.x() * GRAPHICS_ITEM_SIZE, pos.y() * GRAPHICS_ITEM_SIZE, otherPos.x() * GRAPHICS_ITEM_SIZE, otherPos.y() * GRAPHICS_ITEM_SIZE, s);
    QGraphicsScene::addItem(connectionItem);

    //register connection
    _connectionItems[cell->getId()][otherCell->getId()] = connectionItem;
    _connectionItems[otherCell->getId()][cell->getId()] = connectionItem;
}

void ShapeUniverse::delConnectionItem (quint64 cellId)
{
    QMap< quint64, CellConnectionGraphicsItem* > items = _connectionItems.take(cellId);
    if( !items.empty() ) {
        foreach(CellConnectionGraphicsItem* conItem, items.values()) {
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
    QList< CellGraphicsItem* > highlightedCells = _highlightedCells.values();
    foreach(CellGraphicsItem* cellItem, highlightedCells) {
        cellItem->setFocusState(CellGraphicsItem::NO_FOCUS);
    }
    _highlightedCells.clear();

    //defocus old energy particles
    QList< EnergyGraphicsItem* > highlightedEs = _highlightedEnergyParticles.values();
    foreach(EnergyGraphicsItem* eItem, highlightedEs) {
        eItem->setFocusState(EnergyGraphicsItem::NO_FOCUS);
    }
    _highlightedEnergyParticles.clear();
}

void ShapeUniverse::highlightCell (Cell* cell)
{
    if( !cell )
        return;

    //focus cellcluster
    foreach(Cell* otherCell, cell->getCluster()->getCellsRef()) {
        if( _cellItems.contains(otherCell->getId()) ) {
            CellGraphicsItem* cellItem = _cellItems[otherCell->getId()];
            if( cellItem->getFocusState() == CellGraphicsItem::NO_FOCUS )
                cellItem->setFocusState(CellGraphicsItem::FOCUS_CLUSTER);
            _highlightedCells[otherCell->getId()] = cellItem;
        }
    }

    //focus cell
    if( _cellItems.contains(cell->getId()) )
        _cellItems[cell->getId()]->setFocusState(CellGraphicsItem::FOCUS_CELL);
}

void ShapeUniverse::highlightEnergyParticle (EnergyGraphicsItem* e)
{
    if( !e )
        return;

    //focus energy particle
    e->setFocusState(EnergyGraphicsItem::FOCUS);
    _highlightedEnergyParticles[e->getEnergyParticle()->getId()] = e;
}

void ShapeUniverse::setCellColorFromMetadata ()
{
    //set cell colors
    QMapIterator< quint64, CellGraphicsItem* > it(_cellItems);
    while( it.hasNext() ) {
        it.next();
        CellGraphicsItem* cellItem = it.value();
		Cell* cell = cellItem->getCell();
        cellItem->setColor(cell->getMetadata().color);
    }
}

QVector3D ShapeUniverse::calcCenterOfHighlightedObjects ()
{
    QVector3D center;
    QList< CellGraphicsItem* > cellItems(_highlightedCells.values());
    foreach( CellGraphicsItem* cellItem, cellItems )
        center += QVector3D(cellItem->pos().x(), cellItem->pos().y(), 0.0);
    QList< EnergyGraphicsItem* > eItems(_highlightedEnergyParticles.values());
    foreach( EnergyGraphicsItem* eItem, eItems )
        center += QVector3D(eItem->pos().x(), eItem->pos().y(), 0.0);
    return center/(cellItems.size()+eItems.size())/GRAPHICS_ITEM_SIZE;
}


