#include <qmath.h>

#include "global/servicelocator.h"
#include "global/numbergenerator.h"

#include "model/alienfacade.h"
#include "model/physics/physics.h"
#include "model/modelsettings.h"
#include "model/simulationcontext.h"
#include "model/energyparticlemap.h"
#include "model/cellmap.h"
#include "model/topology.h"
#include "model/simulationparameters.h"

#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "energyparticleimpl.h"


EnergyParticleImpl::EnergyParticleImpl(SimulationContext* context)
	: _context(context)
	, _topology(context->getTopology())
	, _cellMap(context->getCellMap())
	, _energyMap(context->getEnergyParticleMap())
	, _parameters(context->getSimulationParameters())
	, _id(NumberGenerator::getInstance().getInstance().createNewTag())
{

}

EnergyParticleImpl::EnergyParticleImpl(qreal energy, QVector3D pos, QVector3D vel, SimulationContext* context)
	: EnergyParticleImpl(context)
{
	_energy = energy;
	_pos = pos;
	_vel = vel;
}

//return: false = energy is zero
//        cluster is nonzero if particle transforms into cell
bool EnergyParticleImpl::processingMovement(CellCluster*& cluster)
{
	_energyMap->removeParticleIfPresent(_pos, this);
	move();

	if (EnergyParticle* otherEnergy = _energyMap->getParticle(_pos)) {
		collisionWithEnergyParticle(otherEnergy);
		return false;
	}

	//is there a cell at new position?
	Cell* cell(_cellMap->getCell(_pos));
	if (cell) {
		collisionWithCell(cell);
		return false;
	}

	//enough energy for cell transformation?
	qreal p(NumberGenerator::getInstance().random());
	qreal eKin = Physics::kineticEnergy(1, _vel, 0, 0);
	qreal eNew = _energy - (eKin / _parameters->cellMass_Reciprocal);
	if ((eNew >= _parameters->cellMinEnergy) && (p < _parameters->cellTransformationProb)) {

		//look for neighbor cell
		for (int dx = -2; dx < 3; ++dx) {
			for (int dy = -2; dy < 3; ++dy) {
				if (_cellMap->getCell(_pos + QVector3D(dx, dy, 0.0))) {
					_energyMap->setParticle(_pos, this);
					return true;
				}
			}
		}

		//create cell and cluster
		QList< Cell* > cells;
		AlienFacade* facade = ServiceLocator::getInstance().getService<AlienFacade>();
		Cell* c = facade->buildFeaturedCellWithRandomData(eNew, _context);
		cells << c;
		cluster = facade->buildCellCluster(cells, 0.0, _pos, 0.0, _vel, _context);
		_energy = 0;
		_cellMap->setCell(_pos, c);
		CellMetadata meta = c->getMetadata();
		meta.color = _metadata.color;
		c->setMetadata(meta);
		return false;
	}
	else {
		_energyMap->setParticle(_pos, this);
	}
	return true;
}

void EnergyParticleImpl::collisionWithCell(Cell* cell)
{
	cell->setEnergy(cell->getEnergy() + _energy);
	//create token?
	/*            if( (cell->getNumToken(true) < CELL_TOKENSTACKSIZE) && (_energy > MIN_TOKEN_ENERGY) ) {
	Token* token = new Token(_energy, true);
	cell->addToken(token,false);
	}
	else
	cell->setEnergy(cell->getEnergy() + _energy);
	*/
	_energy = 0;
}

void EnergyParticleImpl::collisionWithEnergyParticle(EnergyParticle* otherEnergy)
{
	//particle with most energy inherits color
	if (otherEnergy->getEnergy() < _energy)
		otherEnergy->setMetadata(_metadata);

	otherEnergy->setEnergy(otherEnergy->getEnergy() + _energy);
	_energy = 0;
	otherEnergy->setVelocity((otherEnergy->getVelocity() + _vel) / 2.0);
}

void EnergyParticleImpl::move()
{
	_pos += _vel;
	_topology->correctPosition(_pos);
}

void EnergyParticleImpl::serializePrimitives(QDataStream& stream) const
{
	stream << _energy << _pos << _vel << _id;
}

void EnergyParticleImpl::deserializePrimitives(QDataStream& stream)
{
	stream >> _energy >> _pos >> _vel >> _id;
}

qreal EnergyParticleImpl::getEnergy() const
{
	return _energy;
}

void EnergyParticleImpl::setEnergy(qreal value)
{
	_energy = value;
}

QVector3D EnergyParticleImpl::getPosition() const
{
	return _pos;
}

void EnergyParticleImpl::setPosition(QVector3D value)
{
	_pos = value;
}

QVector3D EnergyParticleImpl::getVelocity() const
{
	return _vel;
}

void EnergyParticleImpl::setVelocity(QVector3D value)
{
	_vel = value;
}

quint64 EnergyParticleImpl::getId() const
{
	return _id;
}

void EnergyParticleImpl::setId(quint64 value)
{
	_id = value;
}

EnergyParticleMetadata EnergyParticleImpl::getMetadata() const
{
	return _metadata;
}

void EnergyParticleImpl::setMetadata(EnergyParticleMetadata value)
{
	_metadata = value;
}
