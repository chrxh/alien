#include "model/context/SpaceMetric.h"
#include "model/entities/EnergyParticle.h"

#include "EnergyParticleMapImpl.h"

EnergyParticleMapImpl::EnergyParticleMapImpl(QObject* parent)
	: EnergyParticleMap(parent)
{
}

EnergyParticleMapImpl::~EnergyParticleMapImpl()
{
	deleteGrid();
}


void EnergyParticleMapImpl::init(SpaceMetric* metric, MapCompartment* compartment)
{
	_metric = metric;
	_compartment = compartment;
	deleteGrid();
	_size = _compartment->getSize();
	_energyGrid = new EnergyParticle**[_size.x];
	for (int x = 0; x < _size.x; ++x) {
		_energyGrid[x] = new EnergyParticle*[_size.y];
	}
	clear();
}

void EnergyParticleMapImpl::clear()
{
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y)
			_energyGrid[x][y] = nullptr;
}

void EnergyParticleMapImpl::removeParticleIfPresent(QVector3D pos, EnergyParticle * particleToRemove)
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	EnergyParticle*& particle = locateParticle(intPos);
	if (particle == particleToRemove) {
		particle = nullptr;
	}
}

void EnergyParticleMapImpl::setParticle(QVector3D pos, EnergyParticle * particle)
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	locateParticle(intPos) = particle;
}

EnergyParticle * EnergyParticleMapImpl::getParticle(QVector3D pos) const
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	return locateParticle(intPos);
}

void EnergyParticleMapImpl::serializePrimitives(QDataStream & stream) const
{
	//determine number of energy particle entries
	quint32 numEntries = 0;
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y)
			if (_energyGrid[x][y])
				numEntries++;
	stream << numEntries;

	//write energy particle entries
	for (qint32 x = 0; x < _size.x; ++x) {
		for (qint32 y = 0; y < _size.y; ++y) {
			EnergyParticle* e = _energyGrid[x][y];
			if (e) {
				stream << x << y << e->getId();
			}
		}
	}
}

void EnergyParticleMapImpl::deserializePrimitives(QDataStream & stream, QMap<quint64, EnergyParticle*> const & oldIdEnergyMap)
{
	quint32 numEntries = 0;
	qint32 x = 0;
	qint32 y = 0;
	quint64 oldId = 0;
	stream >> numEntries;
	for (quint32 i = 0; i < numEntries; ++i) {
		stream >> x >> y >> oldId;
		EnergyParticle* particle = oldIdEnergyMap[oldId];
		_energyGrid[x][y] = particle;
	}
}

void EnergyParticleMapImpl::deleteGrid()
{
	if (_energyGrid) {
		for (int x = 0; x < _size.x; ++x) {
			delete[] _energyGrid[x];
		}
		delete[] _energyGrid;
	}
}
