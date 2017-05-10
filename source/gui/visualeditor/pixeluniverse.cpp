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

Q_SLOT void PixelUniverse::requestData()
{
	IntVector2D size = _simAccess->getUniverseSize();
	ResolveDescription resolveDesc;
	_simAccess->requireData({ {0, 0}, {size.x/4 - 1, size.y/4 - 1} }, resolveDesc);
}

Q_SLOT void PixelUniverse::retrieveAndDisplayData()
{
	auto const& dataDesc = _simAccess->retrieveData();

	_image->fill(UNIVERSE_COLOR);
	displayClusters(dataDesc);
	displayparticles(dataDesc);
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

namespace
{
	uint32_t calcCellColor(CellMetadata const& meta, double energy)
	{
		uint8_t r = 0;
		uint8_t g = 0;
		uint8_t b = 0;
		auto const& color = meta.color;
		if (color == 0) {
			r = INDIVIDUAL_CELL_COLOR1.red();
			g = INDIVIDUAL_CELL_COLOR1.green();
			b = INDIVIDUAL_CELL_COLOR1.blue();
		}
		if (color == 1) {
			r = INDIVIDUAL_CELL_COLOR2.red();
			g = INDIVIDUAL_CELL_COLOR2.green();
			b = INDIVIDUAL_CELL_COLOR2.blue();
		}
		if (color == 2) {
			r = INDIVIDUAL_CELL_COLOR3.red();
			g = INDIVIDUAL_CELL_COLOR3.green();
			b = INDIVIDUAL_CELL_COLOR3.blue();
		}
		if (color == 3) {
			r = INDIVIDUAL_CELL_COLOR4.red();
			g = INDIVIDUAL_CELL_COLOR4.green();
			b = INDIVIDUAL_CELL_COLOR4.blue();
		}
		if (color == 4) {
			r = INDIVIDUAL_CELL_COLOR5.red();
			g = INDIVIDUAL_CELL_COLOR5.green();
			b = INDIVIDUAL_CELL_COLOR5.blue();
		}
		if (color == 5) {
			r = INDIVIDUAL_CELL_COLOR6.red();
			g = INDIVIDUAL_CELL_COLOR6.green();
			b = INDIVIDUAL_CELL_COLOR6.blue();
		}
		if (color == 6) {
			r = INDIVIDUAL_CELL_COLOR7.red();
			g = INDIVIDUAL_CELL_COLOR7.green();
			b = INDIVIDUAL_CELL_COLOR7.blue();
		}
		quint32 e = energy / 2.0 + 20.0;
		if (e > 150) {
			e = 150;
		}
		r = r*e / 150;
		g = g*e / 150;
		b = b*e / 150;
		return (r << 16) | (g << 8) | b;
	}

	uint32_t calcParticleColor(double energy)
	{
		quint32 e = (energy + 10)*5;
		if (e > 150) {
			e = 150;
		}
		return (e << 16) | 0x30;
	}
}

void PixelUniverse::displayClusters(DataDescription const& data) const
{
	for (auto const& clusterTracker : data.clusters) {
		auto const& clusterDesc = clusterTracker.getValue();
		for (auto const& cellTracker : clusterDesc.cells) {
			auto const& cellDesc = cellTracker.getValue();
			auto const& pos = cellDesc.pos.getValue();
			auto const& meta = cellDesc.metadata.getValue();
			auto const& energy = cellDesc.energy.getValue();
			_image->setPixel(pos.x(), pos.y(), calcCellColor(meta, energy));
		}
	}
}

void PixelUniverse::displayparticles(DataDescription const & data) const
{
	for (auto const& particleTracker : data.particles) {
		auto const& particleDesc = particleTracker.getValue();
		auto const& pos = particleDesc.pos.getValue();
		auto const& energy = particleDesc.energy.getValue();
		_image->setPixel(pos.x(), pos.y(), calcParticleColor(energy));
	}
}
