#include "editorvisualizer.h"
#include "../entities/aliencellcluster.h"
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsSceneMouseEvent>

EditorVisualizer::EditorVisualizer(QGraphicsScene* editorScene, QObject* parent)
    : Visualizer(parent), _editorScene(editorScene), _space(0), _zoom(20), _focusItem(0), _focusItemMovable(false)
{
    _editorScene->setBackgroundBrush(QBrush(QColor(0,0,0x30)));
    _editorScene->installEventFilter(this);
}

void EditorVisualizer::init (QList< AlienCellCluster* >* clusters, AlienSpace* space)
{
    _clusters = clusters;
    _space = space;
}

void EditorVisualizer::visualize ()
{
    _space->lockData();
    QGraphicsEllipseItem* item(0);
    _focusItem = 0;
    _focusItemMovable = false;
    _cellItems.clear();

    _editorScene->clear();
    foreach( AlienCellCluster* cluster, *_clusters ) {
        foreach( AlienCell* cell, cluster->getCells()) {
            QVector3D pos(cluster->getCoordinate(cell));

            //create circle
            if( cell->tokenPresent() )
                item = _editorScene->addEllipse(pos.x()*_zoom+_zoom/6, pos.y()*_zoom+_zoom/6, _zoom*2/3, _zoom*2/3, QPen(QColor(0xD0,0xD0,0x80)), QBrush(QColor(0xFF,0xFF,0xAA,0xB0)));
            else
                item = _editorScene->addEllipse(pos.x()*_zoom+_zoom/6, pos.y()*_zoom+_zoom/6, _zoom*2/3, _zoom*2/3, QPen(QColor(0x00,0x50,0x10)), QBrush(QColor(0,0xC0,0x30,0xB0)));

            //create line connections
            for(int i = 0; i < 4; ++i ) {
                AlienCell* otherCell(cell->getBoundaryCell(i));
                if( otherCell ) {

                    //otherCell already drawn?
                    if( _cellItems.contains(otherCell->getId()) ) {

                        //find line connection
                        int j(0);
                        while( otherCell->getBoundaryCell(j) != cell) {
                            ++j;
                        };
                        QGraphicsEllipseItem* item2(_cellItems[otherCell->getId()]);
                        QGraphicsLineItem* line((QGraphicsLineItem*)item2->data(j).value<void*>());
                        item->setData(i, QVariant::fromValue((void*)line));
                        line->setData(1, QVariant::fromValue((void*)item));
                    }
                    else {
                        QVector3D otherPos(cluster->getCoordinate(otherCell));
                        QGraphicsLineItem* line(_editorScene->addLine(pos.x()*_zoom+_zoom/2, pos.y()*_zoom+_zoom/2, otherPos.x()*_zoom+_zoom/2, otherPos.y()*_zoom+_zoom/2, QPen(QColor(0x00,0xF0,0x00,0xB0))));
                        item->setData(i, QVariant::fromValue((void*)line));
                        line->setData(0, QVariant::fromValue((void*)item));
                    }
                }
                else {
                    item->setData(i, QVariant(0));
                }
            }
            _cellItems[cell->getId()] = item;
        }
    }
    /*QMapIterator< quint64, QGraphicsEllipseItem* > it(_cellItems);
    while (it.hasNext()) {
        it.next();
        delete it.value();
    }
    _cellItems = cellItems2;*/
    _space->unlockData();
    _editorScene->update();
}

bool  EditorVisualizer::eventFilter(QObject *obj, QEvent *event)
{
    //mouse botton clicked on cell?
    if( event->type() == QEvent::GraphicsSceneMousePress ) {
        QGraphicsSceneMouseEvent* e((QGraphicsSceneMouseEvent*)event);
        QList< QGraphicsItem* > items(_editorScene->items(e->scenePos()));

        //reset item
        if( _focusItem ) {
            _focusItem->setBrush(QColor(0,0xC0,0x30,0xB0));
            QRectF rect(_focusItem->rect());
            rect.setX(rect.x()+_zoom/6);
            rect.setY(rect.y()+_zoom/6);
            rect.setHeight(_zoom*2/3);
            rect.setWidth(_zoom*2/3);
            _focusItem->setRect(rect);
        }
        _focusItem = 0;

        //focus new item?
        foreach(QGraphicsItem* item, items ) {
            if( item->type() == 4) {
                _focusItemMovable = true;
                _focusItem = ((QGraphicsEllipseItem*)item);
                _focusItem->setBrush(QColor(0xB0,0xFF,0xB0,0xE0));
                _focusRelPos = e->scenePos()-_focusItem->rect().topLeft();
                QPointF pos(_focusItem->rect().topLeft());
                _focusItem->setRect(pos.x()-_zoom/6, pos.y()-_zoom/6, _zoom, _zoom);
                _focusItemMovable = true;
            }
        }
    }
    if( event->type() == QEvent::GraphicsSceneWheel )
        _focusItemMovable = false;

    //mouse moved
    if( (event->type() == QEvent::GraphicsSceneMouseMove) && (_focusItem) && _focusItemMovable ) {
        QGraphicsSceneMouseEvent* e((QGraphicsSceneMouseEvent*)event);
        QPointF pos(e->scenePos());

        //update circle
        _focusItem->setRect(pos.x()-_focusRelPos.x()-_zoom/6, pos.y()-_focusRelPos.y()-_zoom/6, _zoom, _zoom);

        //update lines
        for(int i = 0; i<4; ++i ) {
            QVariant data(_focusItem->data(i));
            if( data.type() == QMetaType::VoidStar ) {
                QGraphicsLineItem* line((QGraphicsLineItem*)data.value<void*>());
                QLineF lineCoordinates(line->line());

                //which end?
                if(line->data(0).value<void*>() == ((void*)_focusItem) )
                    lineCoordinates.setP1(QPointF(pos.x()+_zoom/2-_zoom/6, pos.y()+_zoom/2-_zoom/6)-_focusRelPos);
                else
                    lineCoordinates.setP2(QPointF(pos.x()+_zoom/2-_zoom/6, pos.y()+_zoom/2-_zoom/6)-_focusRelPos);
                line->setLine(lineCoordinates);
            }
        }
    }

    //mouse released
    if( event->type() == QEvent::GraphicsSceneMouseRelease ) {
        _focusItemMovable = false;
//        if( _focusItem )
//            _focusItem->setBrush(QColor(0,0xC0,0x30,0xB0));
//        _focusItem = 0;
    }
    return false;
}
