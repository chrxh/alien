#include "universepixelscene.h"
#include "../../simulation/entities/aliengrid.h"
#include "../../simulation/entities/aliencell.h"
#include "../../simulation/entities/aliencellcluster.h"

#include <QImage>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>

UniversePixelScene::UniversePixelScene(QGraphicsScene* pixelScene, QObject* parent)
    : QObject(parent), _pixelMap(pixelScene->addPixmap(QPixmap())), _space(0), _image(0)
{
    pixelScene->setBackgroundBrush(QBrush(QColor(0,0,0x00)));
    _pixelMap->setScale(2.0);
    pixelScene->update();
}

UniversePixelScene::~UniversePixelScene()
{
    if( _image )
        delete _image;
}

void UniversePixelScene::init (AlienGrid* space)
{
    _space = space;
    _image = new QImage(space->getSizeX(), space->getSizeY(), QImage::Format_RGB32);
}

void UniversePixelScene::visualize ()
{
    _image->fill(0xFF000030);
    _space->lockData();
    int sizeX(_space->getSizeX());
    int sizeY(_space->getSizeY());
    quint32 intensity(0);
    AlienCell* cell(0);
    for(int x = 0; x < sizeX; ++x)
        for(int y = 0; y < sizeY; ++y) {

            //draw energy particle
            AlienEnergy* energy(_space->getEnergyFast(x,y));
            if( energy ) {
                quint32 e(energy->amount+10);
                e *= 5;
                if( e > 150)
                    e = 150;
                _image->setPixel(x, y, (e << 16) | 0x30);
            }

            //draw cell
            AlienCell* cell(_space->getCellFast(x,y));
            if( cell ) {
                cell = _space->getCell(QVector3D(x,y,0.0));
                if(cell->getNumToken() > 0 )
                    _image->setPixel(x, y, 0xFFFFFF);
                else {
//                    _image->setPixel(x, y, (cell->getCluster()->getColor()%100)+155+(((cell->getCluster()->getColor()*2)%256)<<8));
                    quint32 e(cell->getEnergy());
                    if( e > 200)
                        e = 200;
                     _image->setPixel(x, y, (e << 16) | ((e*2/3) << 8) | ((e*2/3) << 0)| 0x30);
                }
            }
        }
    _space->unlockData();
    _pixelMap->setPixmap(QPixmap::fromImage(*_image));
}



