#include "ModelBasic/SpaceProperties.h"
#include "Particle.h"

#include "ParticleMapImpl.h"

ParticleMapImpl::ParticleMapImpl(QObject* parent)
	: ParticleMap(parent)
{
}

ParticleMapImpl::~ParticleMapImpl()
{
	deleteGrid();
}


void ParticleMapImpl::init(SpaceProperties* spaceProp, MapCompartment* compartment)
{
	_spaceProp = spaceProp;
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
	IntVector2D intPos = _spaceProp->correctPositionAndConvertToIntVector(pos);
	Particle*& particle = locateParticle(intPos);
	if (particle == particleToRemove) {
		particle = nullptr;
	}
}

void ParticleMapImpl::setParticle(QVector2D pos, Particle * particle)
{
	IntVector2D intPos = _spaceProp->correctPositionAndConvertToIntVector(pos);
	locateParticle(intPos) = particle;
}

Particle * ParticleMapImpl::getParticle(QVector2D pos) const
{
	IntVector2D intPos = _spaceProp->correctPositionAndConvertToIntVector(pos);
	return locateParticle(intPos);
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
