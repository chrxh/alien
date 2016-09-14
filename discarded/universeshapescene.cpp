#include "universeshapescene.h"
#include "../../simulation/entities/aliencellcluster.h"
#include "../../simulation/entities/alienenergy.h"
#include "../../simulation/entities/aliengrid.h"
#include "../../simulation/coordination/aliensimulator.h"
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>

UniverseShapeScene::UniverseShapeScene(QGraphicsScene* editorScene, QObject* parent)
    : QObject(parent), _editorScene(editorScene), _space(0), _zoom(20), _focusItem(0), _focusItemMovable(false), _lastVertMousePos(0.0)
{
    _editorScene->setBackgroundBrush(QBrush(QColor(0,0,0x30)));
    _editorScene->installEventFilter(this);
}

void UniverseShapeScene::init (AlienSimulator* simulator, AlienGrid* space)
{
    _simulator = simulator;
    _space = space;
}

void UniverseShapeScene::visualize ()
{
    _space->lockData();
    QGraphicsEllipseItem* item(0);
    _cellItems.clear();
    _tokenItems.clear();
    _connectionItems.clear();
    _focusItem = 0;
    _focusItemMovable = false;

    _editorScene->clear();

    _editorScene->setSceneRect(0,0,_space->getSizeX()*_zoom,_space->getSizeY()*_zoom);

    //draw energy particles
    foreach( AlienEnergy* energy, _simulator->getEnergyParticles() ) {
        item = _editorScene->addEllipse(energy->pos.x()*_zoom-_zoom/5, energy->pos.y()*_zoom-_zoom/5, _zoom*2/5, _zoom*2/5, QPen(QBrush(),0), QBrush(ENERGY_COLOR));

        //set link to energy particle
        item->setData(1, QVariant(0));  //0 = energy particle
        item->setData(0, QVariant::fromValue((void*)energy));
    }

    //draw cell clusters
    foreach( AlienCellCluster* cluster, _simulator->getClusters() ) {
        foreach( AlienCell* cell, cluster->getCells()) {
            QVector3D pos(cluster->calcPosition(cell));

            //create cell circle
            if( cell->getNumConnections() == cell->getMaxConnections() )
                item = _editorScene->addEllipse(pos.x()*_zoom-_zoom/3, pos.y()*_zoom-_zoom/3, _zoom*2/3, _zoom*2/3, QPen(QBrush(),0), QBrush(CELL_COLOR));
            else
                item = _editorScene->addEllipse(pos.x()*_zoom-_zoom/3, pos.y()*_zoom-_zoom/3, _zoom*2/3, _zoom*2/3, QPen(QBrush(),0), QBrush(CELL_CONNECTABLE_COLOR));

            //set type and link to AlienCell
            item->setData(1, QVariant(1));  //1 = cell
            item->setData(0, QVariant::fromValue((void*)cell));

            //create line connections
            for(int i = 0; i < cell->getNumConnections(); ++i ) {
                AlienCell* otherCell(cell->getConnection(i));

                //otherCell not already drawn?
                if( !_cellItems.contains(otherCell->getId()) ) {

                    //create connection
                    createConnectionItem(cell, otherCell);
                }

            }
            _cellItems[cell->getId()] = item;


            //create token circle
            if( cell->getNumToken() > 0 ) {
                item = _editorScene->addEllipse(pos.x()*_zoom-_zoom/6, pos.y()*_zoom-_zoom/6, _zoom*2/6, _zoom*2/6, QPen(TOKEN_COLOR), QBrush(TOKEN_COLOR));
                _tokenItems[cell->getId()] = item;
            }
        }
    }
//    qDebug("%f,%f - %f,%f",_editorScene->sceneRect().x(),_editorScene->sceneRect().y(),_editorScene->sceneRect().width(),_editorScene->sceneRect().height());
    _space->unlockData();
    _editorScene->update();
}

void UniverseShapeScene::isolateCell (AlienCell* cell)
{
    //delete all lines
    QList< quint64 > ids = _connectionItems[cell->getId()].keys();
    QList< ConnectionItem > values =  _connectionItems[cell->getId()].values();
    foreach( ConnectionItem item, values) {
        delete item.line;
        if( item.lineStart1 ) {
            delete item.lineStart1;
            delete item.lineStart2;
        }
        if( item.lineEnd1 ) {
            delete item.lineEnd1;
            delete item.lineEnd2;
        }
    }
    foreach( quint64 id, ids) {
        _connectionItems[cell->getId()].remove(id);
        _connectionItems[id].remove(cell->getId());
     }
}

void UniverseShapeScene::updateEnergyParticle (AlienEnergy* energy)
{
    QVector3D pos(energy->pos);
    pos = pos * _zoom;
    _focusItem->setRect(pos.x()-_zoom/2.0, pos.y()-_zoom/2.0, _zoom, _zoom);
}

void UniverseShapeScene::updateCluster (AlienCellCluster* cluster)
{
    foreach(AlienCell* cell, cluster->getCells()) {

        //move cell item
        QVector3D pos(cluster->calcPosition(cell));
        QGraphicsEllipseItem* item(_cellItems[cell->getId()]);
        if( item != _focusItem )
            item->setRect(pos.x()*_zoom-_zoom/3, pos.y()*_zoom-_zoom/3, _zoom*2/3, _zoom*2/3);
        else
            item->setRect(pos.x()*_zoom-_zoom/2, pos.y()*_zoom-_zoom/2, _zoom, _zoom);

        //move token item
        if( _tokenItems.contains(cell->getId()) ) {
            item = _tokenItems[cell->getId()];
            item->setRect(pos.x()*_zoom-_zoom/6, pos.y()*_zoom-_zoom/6, _zoom*2/6, _zoom*2/6);
        }

        //move line items
        for( int i = 0; i < cell->getNumConnections(); ++i ) {
            ConnectionItem item(_connectionItems[cell->getId()][cell->getConnection(i)->getId()]);
            moveConnectionItem(cell, cell->getConnection(i), item);
/*            delete item.line;
            if( item.lineStart1 ) {
                delete item.lineStart1;
                delete item.lineStart2;
            }
            if( item.lineEnd1 ) {
                delete item.lineEnd1;
                delete item.lineEnd2;
            }
            createConnectionItem(cell, cell->getConnection(i));*/
        }
    }
}

void UniverseShapeScene::updateFocusCell (AlienCell* cell)
{
    //remove old connections
    isolateCell(cell);

    //draw new connecting line
    for(int i = 0; i < cell->getNumConnections(); ++i) {
        createConnectionItem(cell, cell->getConnection(i));
    }

    //highlight cellcluster
    colorCluster(cell->getCluster());

    //update cell circle
    QVector3D pos(cell->getCluster()->calcPosition(cell));
    pos = pos * _zoom;
    _focusItem->setRect(pos.x()-_zoom/2.0, pos.y()-_zoom/2.0, _zoom, _zoom);

    //update token circle
    if( _tokenItems.contains(cell->getId()) ) {
        _tokenItems[cell->getId()]->setRect(pos.x()-_zoom/6.0, pos.y()-_zoom/6.0, _zoom/3, _zoom/3);
    }

}

void UniverseShapeScene::updateToken ()
{
    if( _focusItem ) {
        AlienCell* cell((AlienCell*)_focusItem->data(0).value<void*>());

        //cell has token
        if( cell->getNumToken() > 0 ) {
            if( !_tokenItems.contains(cell->getId()) ) {

                //create token item
                QVector3D pos(cell->getCluster()->calcPosition(cell));
                QGraphicsEllipseItem* item = _editorScene->addEllipse(pos.x()*_zoom-_zoom/6, pos.y()*_zoom-_zoom/6, _zoom*2/6, _zoom*2/6, QPen(TOKEN_COLOR), QBrush(TOKEN_COLOR));
                _tokenItems[cell->getId()] = item;
            }
        }

        //cell has no token
        else {
            if( _tokenItems.contains(cell->getId()) ) {
                QGraphicsEllipseItem* item(_tokenItems[cell->getId()]);
                delete item;
                _tokenItems.remove(cell->getId());
            }

        }
    }
}

void UniverseShapeScene::focusCell (AlienCell* cell)
{
    _focusItem = _cellItems[cell->getId()];

    //highlight cellcluster
    colorCluster(cell->getCluster());

    //highlight focus item
    QPointF pos(_focusItem->rect().topLeft());
    _focusItem->setRect(pos.x()-_zoom/6, pos.y()-_zoom/6, _zoom, _zoom);

    //SIGNAL: cell clicked
    emit cellClicked(cell);
}

void UniverseShapeScene::focusEnergyParticle (AlienEnergy* energy)
{
    QList< QGraphicsItem* > items(_editorScene->items());
    foreach(QGraphicsItem* item, items ) {
        AlienEnergy* e((AlienEnergy*)item->data(0).value<void*>());
        if( e == energy ) {
            _focusItem = (QGraphicsEllipseItem*)item;
            break;
        }
    }

    if( _focusItem ) {

        //highlight focus item
        QPointF pos(_focusItem->rect().topLeft());
        _focusItem->setRect(pos.x()-_zoom/5, pos.y()-_zoom/5, _zoom*4/5, _zoom*4/5);

        //SIGNAL: cell clicked
        emit energyParticleClicked(energy);
    }
}

bool UniverseShapeScene::eventFilter(QObject *obj, QEvent *event)
{
    //mouse botton clicked on cell?
    if( event->type() == QEvent::GraphicsSceneMousePress ) {
        QGraphicsSceneMouseEvent* e((QGraphicsSceneMouseEvent*)event);

        //reset items
        if( _focusItem ) {

            if( _focusItem->data(1).toInt() == 0 ) {
                //reset size
                QRectF rect(_focusItem->rect());
                rect.setX(rect.x()+_zoom/5);
                rect.setY(rect.y()+_zoom/5);
                rect.setHeight(_zoom*2/5);
                rect.setWidth(_zoom*2/5);
                _focusItem->setRect(rect);
            }

            if( _focusItem->data(1).toInt() == 1 ) {

                //reset cellcluster color
                AlienCell* cell((AlienCell*)_focusItem->data(0).value<void*>());
                decolorCluster(cell->getCluster()->getCells());

                //reset size
                QRectF rect(_focusItem->rect());
                rect.setX(rect.x()+_zoom/6);
                rect.setY(rect.y()+_zoom/6);
                rect.setHeight(_zoom*2/3);
                rect.setWidth(_zoom*2/3);
                _focusItem->setRect(rect);
            }
        }
        _focusItem = 0;

        //focus new item?
        QList< QGraphicsItem* > items(_editorScene->items(e->scenePos()));
        foreach(QGraphicsItem* item, items ) {
            if( (item->type() == 4) && (!item->data(0).isNull()) ) {
                _focusItemMovable = true;

                //energy particle?
                if( (item->data(1).toInt() == 0) ) {
                    _focusItem = ((QGraphicsEllipseItem*)item);

                    _focusRelPos = QVector3D(e->scenePos().x()-_focusItem->rect().topLeft().x(), e->scenePos().y()-_focusItem->rect().topLeft().y(), 0.0);
                    QPointF pos(_focusItem->rect().topLeft());
                    _focusItem->setRect(pos.x()-_zoom/5, pos.y()-_zoom/5, _zoom*4/5, _zoom*4/5);

                    //SIGNAL: energy particle clicked
                    AlienEnergy* energy((AlienEnergy*)_focusItem->data(0).value<void*>());
                    emit energyParticleClicked(energy);
                    break;
                }

                //cell?
                if( (item->data(1).toInt() == 1) ) {
                    _focusItem = ((QGraphicsEllipseItem*)item);

                    //highlight cellcluster
                    AlienCell* cell((AlienCell*)_focusItem->data(0).value<void*>());
                    colorCluster(cell->getCluster());

                    //highlight focus item
                    _focusRelPos = QVector3D(e->scenePos().x()-_focusItem->rect().topLeft().x()-_zoom/3, e->scenePos().y()-_focusItem->rect().topLeft().y()-_zoom/3, 0.0);
                    QPointF pos(_focusItem->rect().topLeft());
                    _focusItem->setRect(pos.x()-_zoom/6, pos.y()-_zoom/6, _zoom, _zoom);

                    //SIGNAL: cell clicked
                    emit cellClicked(cell);
                    break;
                }
            }
        }

        //SIGNAL: defocus
        if( items.isEmpty() ) {
            emit defocus();
        }
    }
    if( event->type() == QEvent::GraphicsSceneWheel )
        _focusItemMovable = false;

    //mouse moved
    if( (event->type() == QEvent::GraphicsSceneMouseMove) && (_focusItem) && _focusItemMovable ) {
        QGraphicsSceneMouseEvent* e((QGraphicsSceneMouseEvent*)event);

        QVector3D pos(e->scenePos().x()-_focusRelPos.x(), e->scenePos().y()-_focusRelPos.y(), 0.0);

        //energy particle?
        if( _focusItem->data(1).toInt() == 0 ) {
            AlienEnergy* energy((AlienEnergy*)_focusItem->data(0).value<void*>());

            //move particle
            energy->pos = pos/_zoom;
            _focusItem->setRect(pos.x()-_zoom/5.0, pos.y()-_zoom/5.0, _zoom*4/5, _zoom*4/5);

            //SIGNAL: energy particle moved
            emit energyParticleMoved();
        }

        //cell?
        if( _focusItem->data(1).toInt() == 1 ) {
            AlienCell* cell((AlienCell*)_focusItem->data(0).value<void*>());
            AlienCellCluster* cluster(cell->getCluster());

            //right button?
            if( e->buttons() == Qt::RightButton ) {

                //move cluster
                pos = pos/_zoom - cluster->calcPosition(cell, _space)+cluster->getPosition();
                _simulator->updateCluster(cluster, pos, cluster->getAngle(),
                                          cluster->getVel(), cluster->getAngularVel());
                updateCluster(cluster);

                //SIGNAL: cluster moved
                emit clusterMoved();
            }
            //left button?
            else if (e->buttons() == Qt::LeftButton ) {

                //move cell
                isolateCell(cell);
                QList< AlienCell* > oldClusterCells(cluster->getCells());
                _simulator->updateCell(cell, QVector3D(pos.x()/_zoom, pos.y()/_zoom, 0));
                decolorCluster(oldClusterCells);
                updateFocusCell(cell);

                //SIGNAL: cell moved
                emit cellMoved();
            }
            //both buttons?
            else {
                //save vertical mouse position
                if( _lastVertMousePos == 0.0 ) {
                    _lastVertMousePos = e->scenePos().y();
                }

                //rotate cluster
                qreal angleInc(e->scenePos().y() - _lastVertMousePos);
                _lastVertMousePos = e->scenePos().y();
                _simulator->updateCluster(cluster, cluster->getPosition(), cluster->getAngle()+angleInc,
                                          cluster->getVel(), cluster->getAngularVel());
                updateCluster(cluster);

                //SIGNAL: cluster moved
                emit clusterMoved();

            }
        }
    }

    //mouse released
    if( event->type() == QEvent::GraphicsSceneMouseRelease ) {
        _focusItemMovable = false;
        _lastVertMousePos = 0.0;
    }

    return false;
}

void UniverseShapeScene::drawArrow (QVector3D p1, QVector3D p2, QGraphicsLineItem*& line1, QGraphicsLineItem*& line2)
{
    QVector3D relPos(p1-p2);
    relPos.normalize();

    //rotate 45 degree counterclockwise
    QVector3D a(relPos.x()-relPos.y(), relPos.x()+relPos.y(), 0);
    a = a / 10.0;
    QVector3D b(p2+relPos*0.35);
    line1 = _editorScene->addLine(b.x()*_zoom, b.y()*_zoom, (b+a).x()*_zoom, (b+a).y()*_zoom, QPen(LINE_ACTIVE_COLOR));

    //rotate 45 degree clockwise
    a = QVector3D(relPos.x()+relPos.y(), -relPos.x()+relPos.y(), 0);
    a = a / 10.0;
    line2 = _editorScene->addLine(b.x()*_zoom, b.y()*_zoom, (b+a).x()*_zoom, (b+a).y()*_zoom, QPen(LINE_ACTIVE_COLOR));
}

void UniverseShapeScene::moveArrow (QVector3D p1, QVector3D p2, QGraphicsLineItem* line1, QGraphicsLineItem* line2)
{
    QVector3D relPos(p1-p2);
    relPos.normalize();

    //rotate 45 degree counterclockwise
    QVector3D a(relPos.x()-relPos.y(), relPos.x()+relPos.y(), 0);
    a = a / 10.0;
    QVector3D b(p2+relPos*0.35);
    if( line1 )
        line1->setLine(b.x()*_zoom, b.y()*_zoom, (b+a).x()*_zoom, (b+a).y()*_zoom);

    //rotate 45 degree clockwise
    a = QVector3D(relPos.x()+relPos.y(), -relPos.x()+relPos.y(), 0);
    a = a / 10.0;
    if( line2 )
        line2->setLine(b.x()*_zoom, b.y()*_zoom, (b+a).x()*_zoom, (b+a).y()*_zoom);
}

ConnectionItem UniverseShapeScene::createConnectionItem (AlienCell* cell, AlienCell* otherCell)
{
    QVector3D pos(cell->getCluster()->calcPosition(cell));
    QVector3D otherPos(otherCell->getCluster()->calcPosition(otherCell));

    //draw line
    QVector3D relPos(otherPos-pos);
    relPos.normalize();
    ConnectionItem connection;
    qreal x1((pos.x()+relPos.x()*0.35)*_zoom);
    qreal y1((pos.y()+relPos.y()*0.35)*_zoom);
    qreal x2((otherPos.x()-relPos.x()*0.35)*_zoom);
    qreal y2((otherPos.y()-relPos.y()*0.35)*_zoom);
    if( (((cell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber() && (!otherCell->blockToken()))
          || (cell->getTokenAccessNumber() == ((otherCell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) && (!cell->blockToken())) )
        connection.line = _editorScene->addLine(x1, y1, x2, y2, QPen(LINE_ACTIVE_COLOR));
    else
        connection.line = _editorScene->addLine(x1, y1, x2, y2, QPen(LINE_INACTIVE_COLOR));

    //draw arrows
    if( ((cell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber() && (!otherCell->blockToken()) )
        drawArrow(pos, otherPos, connection.lineEnd1, connection.lineEnd2);
    if( cell->getTokenAccessNumber() == ((otherCell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) && (!cell->blockToken()))
        drawArrow(otherPos, pos, connection.lineStart1, connection.lineStart2);

    //register connection
    _connectionItems[cell->getId()][otherCell->getId()] = connection;
    _connectionItems[otherCell->getId()][cell->getId()] = connection;
    return connection;
}

void UniverseShapeScene::moveConnectionItem (AlienCell* cell, AlienCell* otherCell, ConnectionItem item)
{
    QVector3D pos(cell->getCluster()->calcPosition(cell));
    QVector3D otherPos(otherCell->getCluster()->calcPosition(otherCell));

    //move line
    QVector3D relPos(otherPos-pos);
    relPos.normalize();
    ConnectionItem connection;
    qreal x1((pos.x()+relPos.x()*0.35)*_zoom);
    qreal y1((pos.y()+relPos.y()*0.35)*_zoom);
    qreal x2((otherPos.x()-relPos.x()*0.35)*_zoom);
    qreal y2((otherPos.y()-relPos.y()*0.35)*_zoom);
    item.line->setLine(x1, y1, x2, y2);

    //move arrows
    if( ((cell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) == otherCell->getTokenAccessNumber() ) {
        moveArrow(pos, otherPos, item.lineEnd1, item.lineEnd2);
    }
    if( cell->getTokenAccessNumber() == ((otherCell->getTokenAccessNumber()+1)%MAX_TOKEN_ACCESS_NUMBERS) )
        moveArrow(otherPos, pos, item.lineStart1, item.lineStart2);
}

void UniverseShapeScene::colorCluster (AlienCellCluster* cluster)
{
    //highlight cellcluster
    foreach(AlienCell* otherCell, cluster->getCells()) {
        QGraphicsEllipseItem* otherItem(_cellItems[otherCell->getId()]);
        if( otherCell->getNumConnections() == otherCell->getMaxConnections() )
            otherItem->setBrush(CELL_CLUSTER_FOCUS_COLOR);
        else
            otherItem->setBrush(CELL_CLUSTER_CONNECTABLE_FOCUS_COLOR);
    }
}

void UniverseShapeScene::decolorCluster (QList< AlienCell* >& cells)
{
    //reset cellcluster color
    foreach(AlienCell* otherCell, cells) {
        QGraphicsEllipseItem* otherItem(_cellItems[otherCell->getId()]);
        if( otherCell->getNumConnections() == otherCell->getMaxConnections() )
            otherItem->setBrush(CELL_COLOR);
        else
            otherItem->setBrush(CELL_CONNECTABLE_COLOR);
    }
}

QVector3D UniverseShapeScene::getViewCenter ()
{
    QGraphicsView* view(_editorScene->views().first());
    QPointF p(view->mapToScene(view->width()/2, view->height()/2));
    return QVector3D(p.x()/_zoom, p.y()/_zoom, 0.0);
}

