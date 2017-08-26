#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "gui/Settings.h"
#include "gui/visualeditor/ViewportInterface.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/SimulationController.h"
#include "Model/Context/SimulationContextApi.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/SpaceMetricApi.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Particle.h"

#include "PixelUniverse.h"

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

void PixelUniverse::init(SimulationController* controller, SimulationAccess* access, ViewportInterface* viewport)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_simAccess = access;

	delete _image;
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_image = new QImage(size.x, size.y, QImage::Format_RGB32);
	QGraphicsScene::setSceneRect(0, 0, _image->width(), _image->height());

}

void PixelUniverse::activate()
{
	connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverse::requestData);
	connect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverse::retrieveAndDisplayData, Qt::QueuedConnection);

	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_simAccess->requireImage({ { 0, 0 }, size }, _image);
}

void PixelUniverse::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverse::requestData);
	disconnect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverse::retrieveAndDisplayData);
}

void PixelUniverse::requestData()
{
/*
	ResolveDescription resolveDesc;
	IntRect rect = _viewport->getRect();
	_simAccess->requireData(rect, resolveDesc);
*/
	IntRect rect = _viewport->getRect();
	_simAccess->requireImage(rect, _image);
}

void PixelUniverse::retrieveAndDisplayData()
{
/*
	auto const& data = _simAccess->retrieveData();

	_image->fill(UNIVERSE_COLOR);

	displayClusters(data);
	displayParticles(data);
*/
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
	auto space = _controller->getContext()->getSpaceMetric();
	for (auto const& cluster : data.clusters) {
		for (auto const& cell : cluster.cells) {
			auto const& pos = *cell.pos;
			auto const& meta = *cell.metadata;
			auto const& energy = *cell.energy;
			auto intPos = space->correctPositionAndConvertToIntVector(pos);
			_image->setPixel(intPos.x, intPos.y, calcCellColor(meta, energy));
		}
	}
}

void PixelUniverse::displayParticles(DataDescription const & data) const
{
	auto space = _controller->getContext()->getSpaceMetric();
	for (auto const& particle: data.particles) {
		auto const& pos = *particle.pos;
		auto const& energy = *particle.energy;
		auto intPos = space->correctPositionAndConvertToIntVector(pos);
		_image->setPixel(intPos.x, intPos.y, calcParticleColor(energy));
	}
}
