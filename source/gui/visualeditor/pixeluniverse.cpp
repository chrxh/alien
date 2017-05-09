#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "global/ServiceLocator.h"
#include "gui/GuiSettings.h"
#include "model/AccessPorts/SimulationAccess.h"
#include "model/BuilderFacade.h"
#include "model/SimulationController.h"
#include "model/context/SimulationContext.h"
#include "model/context/EnergyParticleMap.h"
#include "model/context/CellMap.h"
#include "model/context/SpaceMetric.h"
#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EnergyParticle.h"

#include "pixeluniverse.h"

const int MOUSE_HISTORY = 10;

PixelUniverse::PixelUniverse(QObject* parent)
{
	setBackgroundBrush(QBrush(BACKGROUND_COLOR));
    _pixmap = addPixmap(QPixmap());
    update();
}

PixelUniverse::~PixelUniverse()
{
	delete _image;
}

void PixelUniverse::init(SimulationController* controller)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	_simAccess = facade->buildSimulationAccess(controller->getContext());

	IntVector2D size = _simAccess->getUniverseSize();
	if (!_image) {
		_image = new QImage(size.x, size.y, QImage::Format_RGB32);
		setSceneRect(0, 0, _image->width(), _image->height());
	}

	connect(controller, &SimulationController::timestepCalculated, this, &PixelUniverse::requestData);
	connect(_simAccess, &SimulationAccess::dataReadyToRetrieve, this, &PixelUniverse::retrieveAndDisplayData);

	requestData();
}

void PixelUniverse::reset ()
{
    delete _image;
    _image = nullptr;
    update();
}

/*void PixelUniverse::universeUpdated (SimulationContext* context)
{
	_context = context;

    //prepare image
	context->lock();
	IntVector2D size = context->getSpaceMetric()->getSize();
	context->unlock();
    if( !_image ) {
        _image = new QImage(size.x, size.y, QImage::Format_RGB32);
        setSceneRect(0,0,_image->width(), _image->height());
    }
    _image->fill(0xFF000030);

    //draw image
	context->lock();
    quint8 r = 0;
    quint8 g = 0;
    quint8 b = 0;
	EnergyParticleMap* energyMap = context->getEnergyParticleMap();
	CellMap* cellMap = context->getCellMap();
	IntVector2D pos{ 0, 0 };
    for(pos.x = 0; pos.x < size.x; ++pos.x)
        for(pos.y = 0; pos.y < size.y; ++pos.y) {

            //draw energy particle
			EnergyParticle* energy(energyMap->getParticleFast(pos));
            if( energy ) {
                quint32 e(energy->getEnergy()+10);
                e *= 5;
                if( e > 150)
                    e = 150;
                _image->setPixel(pos.x, pos.y, (e << 16) | 0x30);
            }

            //draw cell
            Cell* cell = cellMap->getCellFast(pos);
            if( cell ) {
                if(cell->getNumToken() > 0 )
                    _image->setPixel(pos.x, pos.y, 0xFFFFFF);
                else {
                    quint8 color = cell->getMetadata().color;
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
                    _image->setPixel(pos.x, pos.y, (r << 16) | (g << 8) | b);
                }
            }
        }

    //draw selection markers
    if( !_selectedClusters.empty() ) {
        for(int x = 0; x < size.x; ++x)
            _image->setPixel(x, _selectionPos.y(), 0x202040);
        for(int y = 0; y < size.y; ++y)
            _image->setPixel(_selectionPos.x(), y, 0x202040);
    }

    //draw selected clusters
    foreach(CellCluster* cluster, _selectedClusters) {
        foreach(Cell* cell, cluster->getCellsRef()) {
            QVector2D pos = cell->calcPosition(true);
            _image->setPixel(pos.x(), pos.y(), 0xBFBFBF);
        }
    }

    context->unlock();
    _pixelMap->setPixmap(QPixmap::fromImage(*_image));

}*/

Q_SLOT void PixelUniverse::requestData()
{
	IntVector2D size = _simAccess->getUniverseSize();
	_simAccess->requireData({ {0, 0}, {size.x - 1, size.y - 1} });
}

Q_SLOT void PixelUniverse::retrieveAndDisplayData()
{
	auto const& dataDesc = _simAccess->retrieveData();

	_image->fill(UNIVERSE_COLOR);
	displayClusters(dataDesc);
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

void PixelUniverse::displayClusters(DataDescription const& data) const
{
	for (auto const& clusterTracker : data.clusters) {
		auto const& clusterDesc = clusterTracker.getValue();
		for (auto const& cellTracker : clusterDesc.cells) {
			auto const& cellDesc = cellTracker.getValue();
			auto const& pos = cellDesc.pos.getValue();
			_image->setPixel(pos.x(), pos.y(), 0xFFFFFF);
		}
	}
}
