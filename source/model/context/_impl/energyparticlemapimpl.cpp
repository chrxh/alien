#include "model/context/topology.h"
#include "model/entities/energyparticle.h"

#include "energyparticlemapimpl.h"

EnergyParticleMapImpl::EnergyParticleMapImpl(QObject* parent)
	: EnergyParticleMap(parent)
{
}

EnergyParticleMapImpl::~EnergyParticleMapImpl()
{
	deleteGrid();
}


void EnergyParticleMapImpl::init(Topology* topo, MapCompartment* compartment)
{
	_topo = topo;
	deleteGrid();
	IntVector2D size = _topo->getSize();
	_gridSize = size.x;
	_energyGrid = new EnergyParticle**[size.x];
	for (int x = 0; x < size.x; ++x) {
		_energyGrid[x] = new EnergyParticle*[size.y];
	}
	clear();
}

void EnergyParticleMapImpl::clear()
{
	IntVector2D size = _topo->getSize();
	for (int x = 0; x < size.x; ++x)
		for (int y = 0; y < size.y; ++y)
			_energyGrid[x][y] = nullptr;
}

void EnergyParticleMapImpl::removeParticleIfPresent(QVector3D pos, EnergyParticle * energy)
{
	IntVector2D intPos = _topo->correctPositionWithIntPrecision(pos);
	if (_energyGrid[intPos.x][intPos.y] == energy)
		_energyGrid[intPos.x][intPos.y] = nullptr;
}

void EnergyParticleMapImpl::setParticle(QVector3D pos, EnergyParticle * energy)
{
	IntVector2D intPos = _topo->correctPositionWithIntPrecision(pos);
	_energyGrid[intPos.x][intPos.y] = energy;
}

EnergyParticle * EnergyParticleMapImpl::getParticle(QVector3D pos) const
{
	IntVector2D intPos = _topo->correctPositionWithIntPrecision(pos);
	return _energyGrid[intPos.x][intPos.y];
}

void EnergyParticleMapImpl::serializePrimitives(QDataStream & stream) const
{
	//determine number of energy particle entries
	quint32 numEntries = 0;
	IntVector2D size = _topo->getSize();
	for (int x = 0; x < size.x; ++x)
		for (int y = 0; y < size.y; ++y)
			if (_energyGrid[x][y])
				numEntries++;
	stream << numEntries;

	//write energy particle entries
	for (qint32 x = 0; x < size.x; ++x)
		for (qint32 y = 0; y < size.y; ++y) {
			EnergyParticle* e = _energyGrid[x][y];
			if (e) {
				stream << x << y << e->getId();
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
		for (int x = 0; x < _gridSize; ++x) {
			delete[] _energyGrid[x];
		}
		delete[] _energyGrid;
	}
}
