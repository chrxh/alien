#include "Model/Local/SpaceMetricLocal.h"
#include "Model/Local/Particle.h"

#include "ParticleMapImpl.h"

ParticleMapImpl::ParticleMapImpl(QObject* parent)
	: ParticleMap(parent)
{
}

ParticleMapImpl::~ParticleMapImpl()
{
	deleteGrid();
}


void ParticleMapImpl::init(SpaceMetricLocal* metric, MapCompartment* compartment)
{
	_metric = metric;
	_compartment = compartment;

	deleteGrid();
	_size = _compartment->getSize();
	_energyGrid = new Particle**[_size.x];
	for (int x = 0; x < _size.x; ++x) {
		_energyGrid[x] = new Particle*[_size.y];
	}
	clear();
}

void ParticleMapImpl::clear()
{
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y)
			_energyGrid[x][y] = nullptr;
}

void ParticleMapImpl::removeParticleIfPresent(QVector2D pos, Particle * particleToRemove)
{
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	Particle*& particle = locateParticle(intPos);
	if (particle == particleToRemove) {
		particle = nullptr;
	}
}

void ParticleMapImpl::setParticle(QVector2D pos, Particle * particle)
{
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	locateParticle(intPos) = particle;
}

Particle * ParticleMapImpl::getParticle(QVector2D pos) const
{
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	return locateParticle(intPos);
}

void ParticleMapImpl::serializePrimitives(QDataStream & stream) const
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
			Particle* e = _energyGrid[x][y];
			if (e) {
				stream << x << y << e->getId();
			}
		}
	}
}

void ParticleMapImpl::deserializePrimitives(QDataStream & stream, QMap<quint64, Particle*> const & oldIdEnergyMap)
{
	quint32 numEntries = 0;
	qint32 x = 0;
	qint32 y = 0;
	quint64 oldId = 0;
	stream >> numEntries;
	for (quint32 i = 0; i < numEntries; ++i) {
		stream >> x >> y >> oldId;
		Particle* particle = oldIdEnergyMap[oldId];
		_energyGrid[x][y] = particle;
	}
}

void ParticleMapImpl::deleteGrid()
{
	if (_energyGrid) {
		for (int x = 0; x < _size.x; ++x) {
			delete[] _energyGrid[x];
		}
		delete[] _energyGrid;
	}
}
