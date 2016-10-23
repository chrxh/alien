#include "pixeluniverse.h"

#include "../../global/guisettings.h"

#include "../../model/entities/aliengrid.h"
#include "../../model/entities/aliencell.h"
#include "../../model/entities/aliencellcluster.h"
#include "../../model/entities/alienenergy.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>

const int MOUSE_HISTORY = 10;

PixelUniverse::PixelUniverse(QObject* parent)
    : _pixelMap(0), _image(0), _timer(0), _lastMouseDiffs(MOUSE_HISTORY),
      _leftMouseButtonPressed(false), _rightMouseButtonPressed(false)
{
    setBackgroundBrush(QBrush(QColor(0,0,0)));
    _pixelMap = addPixmap(QPixmap());
//    _pixelMap->setScale(_zoom);
    update();

    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timeout()));
    _timer->start(500);
}

PixelUniverse::~PixelUniverse()
{
    if( _image )
        delete _image;
}

void PixelUniverse::reset ()
{
    if( _image )
        delete _image;
    _image = 0;
//    delete _pixelMap;
//    _pixelMap = addPixmap(QPixmap());
    update();
}

void PixelUniverse::universeUpdated (AlienGrid* grid)
{
    _grid = grid;

    //prepare image
    grid->lockData();
    int sizeX(grid->getSizeX());
    int sizeY(grid->getSizeY());
    grid->unlockData();
    if( !_image ) {
//        _pixelMap = addPixmap(QPixmap());
        _image = new QImage(sizeX, sizeY, QImage::Format_RGB32);
        setSceneRect(0,0,_image->width(), _image->height());
    }
    _image->fill(0xFF000030);

    //draw image
    grid->lockData();
    quint8 r = 0;
    quint8 g = 0;
    quint8 b = 0;
    for(int x = 0; x < sizeX; ++x)
        for(int y = 0; y < sizeY; ++y) {

            //draw energy particle
            AlienEnergy* energy(grid->getEnergyFast(x,y));
            if( energy ) {
                quint32 e(energy->amount+10);
                e *= 5;
                if( e > 150)
                    e = 150;
                _image->setPixel(x, y, (e << 16) | 0x30);
            }

            //draw cell
            AlienCell* cell(grid->getCellFast(x,y));
            if( cell ) {
//                cell = grid->getCell(QVector3D(x,y,0.0));
                if(cell->getNumToken() > 0 )
                    _image->setPixel(x, y, 0xFFFFFF);
                else {
                    quint8 color = cell->getColor();
                    if( color == 0 ) {
                        r = INDIVIDUAL_CELL_COLOR1.red();
                        g = INDIVIDUAL_CELL_COLOR1.green();
                        b = INDIVIDUAL_CELL_COLOR1.blue();
                    }
                    if( color == 1 ) {
                        r = INDIVIDUAL_CELL_COLOR2.red();
                        g = INDIVIDUAL_CELL_COLOR2.green();
                        b = INDIVIDUAL_CELL_COLOR2.blue();
                    }
                    if( color == 2 ) {
                        r = INDIVIDUAL_CELL_COLOR3.red();
                        g = INDIVIDUAL_CELL_COLOR3.green();
                        b = INDIVIDUAL_CELL_COLOR3.blue();
                    }
                    if( color == 3 ) {
                        r = INDIVIDUAL_CELL_COLOR4.red();
                        g = INDIVIDUAL_CELL_COLOR4.green();
                        b = INDIVIDUAL_CELL_COLOR4.blue();
                    }
                    if( color == 4 ) {
                        r = INDIVIDUAL_CELL_COLOR5.red();
                        g = INDIVIDUAL_CELL_COLOR5.green();
                        b = INDIVIDUAL_CELL_COLOR5.blue();
                    }
                    if( color == 5 ) {
                        r = INDIVIDUAL_CELL_COLOR6.red();
                        g = INDIVIDUAL_CELL_COLOR6.green();
                        b = INDIVIDUAL_CELL_COLOR6.blue();
                    }
                    if( color == 6 ) {
                        r = INDIVIDUAL_CELL_COLOR7.red();
                        g = INDIVIDUAL_CELL_COLOR7.green();
                        b = INDIVIDUAL_CELL_COLOR7.blue();
                    }
                    quint32 e(cell->getEnergy()/2.0+20.0);
                    if( e > 150)
                        e = 150;
                    r = r*e/150;
                    g = g*e/150;
                    b = b*e/150;
//                    _image->setPixel(x, y, (e << 16) | ((e*2/3) << 8) | ((e*2/3) << 0)| 0x30);
                    _image->setPixel(x, y, (r << 16) | (g << 8) | b);
                }
            }
        }

    //draw selection markers
    if( !_selectedClusters.empty() ) {
        for(int x = 0; x < sizeX; ++x)
            _image->setPixel(x, _selectionPos.y(), 0x202040);
        for(int y = 0; y < sizeY; ++y)
            _image->setPixel(_selectionPos.x(), y, 0x202040);
    }

    //draw selected clusters
    foreach(AlienCellCluster* cluster, _selectedClusters) {
        foreach(AlienCell* cell, cluster->getCells()) {
            QVector3D pos = cell->calcPosition(_grid);
            _image->setPixel(pos.x(), pos.y(), 0xBFBFBF);
        }
    }

    grid->unlockData();
    _pixelMap->setPixmap(QPixmap::fromImage(*_image));
}

void PixelUniverse::mousePressEvent (QGraphicsSceneMouseEvent* e)
{
    if( !_grid )
        return;
    _grid->lockData();

    //update mouse buttons
    _leftMouseButtonPressed = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
    _rightMouseButtonPressed = ((e->buttons() & Qt::RightButton) == Qt::RightButton);

    //left xor right button pressed?
    if( (_leftMouseButtonPressed && (!_rightMouseButtonPressed)) || ((!_leftMouseButtonPressed) && _rightMouseButtonPressed)) {

        //scan for clusters
        QMap< quint64, AlienCellCluster* > clusters;
        QVector3D mousePos(e->scenePos().x(), e->scenePos().y(), 0.0);
        for(int rx = -5; rx < 6; ++rx )
            for(int ry = -5; ry < 6; ++ry ) {
                QVector3D scanPos = mousePos + QVector3D(rx,ry,0.0);
                if( (scanPos.x() >= 0.0) && (scanPos.x() < _grid->getSizeX())
                    && (scanPos.y() >= 0.0) && (scanPos.y() < _grid->getSizeY()) ) {
                    AlienCell* cell = _grid->getCell(scanPos);
                    if( cell)
                        clusters[cell->getCluster()->getId()] = cell->getCluster();
                }
            }

        //remove clusters from simulation (temporarily)
        foreach(AlienCellCluster* cluster, clusters) {
            _grid->getClusters().removeOne(cluster);
            cluster->clearCellsFromMap();
        }

        //calc center
        QVector3D center;
        int numCells = 0;
        foreach(AlienCellCluster* cluster, clusters) {
            foreach(AlienCell* cell, cluster->getCells()) {
                center += cell->calcPosition();
            }
            numCells += cluster->getCells().size();
        }
        center = center / numCells;

        //move to selected clusters
        _selectedClusters = clusters.values();
        _selectionPos.setX(center.x());
        _selectionPos.setY(center.y());
        _grid->correctPosition(_selectionPos);
    }

    //both buttons pressed?
    if( _leftMouseButtonPressed && _rightMouseButtonPressed ) {

        //move selected clusters to simulation
        foreach(AlienCellCluster* cluster, _selectedClusters) {
            cluster->drawCellsToMap();
        }
        _grid->getClusters() << _selectedClusters;
        _selectedClusters.clear();
    }
    _grid->unlockData();
    universeUpdated(_grid);
}

void PixelUniverse::mouseReleaseEvent (QGraphicsSceneMouseEvent* e)
{
    if( !_grid )
        return;
    _grid->lockData();

    //update mouse buttons
    _leftMouseButtonPressed = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
    _rightMouseButtonPressed = ((e->buttons() & Qt::RightButton) == Qt::RightButton);

    //move selected clusters to simulation
    foreach(AlienCellCluster* cluster, _selectedClusters) {
        cluster->drawCellsToMap();
    }
    _grid->getClusters() << _selectedClusters;
    _selectedClusters.clear();

    _grid->unlockData();
    universeUpdated(_grid);
}

void PixelUniverse::mouseMoveEvent (QGraphicsSceneMouseEvent* e)
{
    if( !_grid )
        return;

    //update mouse buttons and positions
//    _leftMouseButtonPressed = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
//    _rightMouseButtonPressed = ((e->buttons() & Qt::RightButton) == Qt::RightButton);
    QVector3D mousePos(e->scenePos().x(), e->scenePos().y(), 0.0);
    QVector3D lastMousePos(e->lastScenePos().x(), e->lastScenePos().y(), 0.0);
    QVector3D mouseDiff = mousePos - lastMousePos;
    QVector3D cumMouseDiff = mouseDiff;
    for(int i = 0; i < MOUSE_HISTORY; ++i) {
        cumMouseDiff += _lastMouseDiffs[i];
    }
    cumMouseDiff = cumMouseDiff / MOUSE_HISTORY;

    //only left button pressed? => move selected clusters
    if( _leftMouseButtonPressed && (!_rightMouseButtonPressed) ) {
        _grid->lockData();

        //update position and velocity
        foreach(AlienCellCluster* cluster, _selectedClusters) {
            cluster->setPosition(cluster->getPosition()+mouseDiff);
            cluster->setVel((cumMouseDiff)/5.0);
        }

        //update selection
        _selectionPos +=mouseDiff;
        _grid->correctPosition(_selectionPos);

        _grid->unlockData();
        universeUpdated(_grid);
    }

    //only right button pressed? => rotate selected clusters
    if( (!_leftMouseButtonPressed) && _rightMouseButtonPressed ) {
        _grid->lockData();

        //1. step: rotate each cluster around own center
        foreach(AlienCellCluster* cluster, _selectedClusters) {
            cluster->setAngle(cluster->getAngle()+mouseDiff.x()+mouseDiff.y());
            cluster->setAngularVel((cumMouseDiff.x() + cumMouseDiff.y())/3.0);
        }

        //2. step: rotate cluster around common center
        //calc center
        QVector3D center;
        int numCells = 0;
        foreach(AlienCellCluster* cluster, _selectedClusters) {
            foreach(AlienCell* cell, cluster->getCells()) {
                center += cell->calcPosition();
            }
            numCells += cluster->getCells().size();
        }
        center = center / numCells;
        QMatrix4x4 transform;
        transform.setToIdentity();
        transform.translate(center);
        transform.rotate(mouseDiff.x()+mouseDiff.y(), 0.0, 0.0, 1.0);
        transform.translate(-center);
        foreach(AlienCellCluster* cluster, _selectedClusters) {
            cluster->setPosition(transform.map(cluster->getPosition()));
        }

        _grid->unlockData();
        universeUpdated(_grid);
    }

    //both buttons pressed? => apply forces along mouse path
    if( _leftMouseButtonPressed && _rightMouseButtonPressed ) {
        if( mousePos != lastMousePos ) {
            _grid->lockData();

            //calc distance vector and length
            QVector3D dir = mouseDiff.normalized();
            qreal dist = mouseDiff.length();

            //scan mouse path for clusters
            QMap< quint64, AlienCellCluster* > clusters;
            QMap< quint64, AlienCell* > cells;
            for(int d = 0; d < qFloor(dist)+1; ++d ) {
                for(int rx = -5; rx < 6; ++rx )
                    for(int ry = -5; ry < 6; ++ry ) {
                        QVector3D scanPos = mousePos + dir*d + QVector3D(rx,ry,0.0);
                        AlienCell* cell = _grid->getCell(scanPos);
                        if( cell) {
                            clusters[cell->getCluster()->getId()] = cell->getCluster();
                            cells[cell->getCluster()->getId()] = cell;
                        }
                    }
            }

            //apply forces to all encountered cells
            QMapIterator< quint64, AlienCell* > itCell(cells);
            while(itCell.hasNext()) {
                itCell.next();
                AlienCell* cell = itCell.value();

                //apply force
                cell->setVel(cell->getVel() + dir*dist*cell->getCluster()->getMass()*0.05);
            }

            //calc effective velocities of the clusters
            QMapIterator< quint64, AlienCellCluster* > itCluster(clusters);
            while(itCluster.hasNext()) {
                itCluster.next();
                AlienCellCluster* cluster = itCluster.value();
                cluster->updateVel_angularVel_via_cellVelocities();
                cluster->updateCellVel(false);
            }
            _grid->unlockData();
        }
    }
    for(int i = 0; i < MOUSE_HISTORY-1; ++i)
        _lastMouseDiffs[MOUSE_HISTORY-i-1] = _lastMouseDiffs[MOUSE_HISTORY-i-2];
    _lastMouseDiffs[0] = mouseDiff;
}

void PixelUniverse::timeout ()
{
    //set velocity of selected clusters to 0
    foreach(AlienCellCluster* cluster, _selectedClusters) {
        if( _leftMouseButtonPressed )
            cluster->setVel(QVector3D());
        if( _rightMouseButtonPressed )
            cluster->setAngularVel(0.0);
    }
}

