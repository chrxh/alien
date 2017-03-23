#include <qmath.h>

#include "global/servicelocator.h"
#include "global/global.h"

#include "model/alienfacade.h"
#include "model/physics/physics.h"
#include "model/config.h"
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
	, _id(GlobalFunctions::createNewTag())
{

}

EnergyParticleImpl::EnergyParticleImpl(qreal energy, QVector3D pos, QVector3D vel, SimulationContext* context)
	: EnergyParticleImpl(context)
{
	_energy = energy;
	_pos = pos;
	_vel = vel;
}

//return: true = energy is zero
//        cluster is nonzero if particle transforms into cell
bool EnergyParticleImpl::movement(CellCluster*& cluster)
{
	//remove old position from map
	_energyMap->removeParticleIfPresent(_pos, this);

	//update position
	_pos += _vel;
	_topology->correctPosition(_pos);

	//apply gravitational force
	/*    QVector3D gSource1(200.0+qSin(0.5*degToRad*(qreal)time)*50, 200.0+qCos(0.5*degToRad*(qreal)time)*50, 0.0);
	QVector3D gSource2(200.0-qSin(0.5*degToRad*(qreal)time)*50, 200.0-qCos(0.5*degToRad*(qreal)time)*50, 0.0);
	QVector3D distance1 = gSource1-pos;
	QVector3D distance2 = gSource1-(pos+vel);
	grid->correctDistance(distance1);
	grid->correctDistance(distance2);
	vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
	vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
	distance1 = gSource2-pos;
	distance2 = gSource2-(pos+vel);
	grid->correctDistance(distance1);
	grid->correctDistance(distance2);
	vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
	vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
	*/
	//is there energy at new position?
	EnergyParticle* otherEnergy(_energyMap->getParticle(_pos));
	if (otherEnergy) {

		//particle with most energy inherits color
		if (otherEnergy->getEnergy() < _energy)
			otherEnergy->setMetadata(_metadata);

		otherEnergy->setEnergy(otherEnergy->getEnergy() + _energy);
		_energy = 0;
		otherEnergy->setVelocity((otherEnergy->getVelocity() + _vel) / 2.0);

		return true;
	}
	else {
		//is there a cell at new position?
		Cell* cell(_cellMap->getCell(_pos));
		if (cell) {
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
			return true;
		}
		else {

			//enough energy for cell transformation?
			qreal p((qreal)qrand() / RAND_MAX);
			qreal eKin = Physics::kineticEnergy(1, _vel, 0, 0);
			qreal eNew = _energy - (eKin / _parameters->INTERNAL_TO_KINETIC_ENERGY);
			if ((eNew >= _parameters->CRIT_CELL_TRANSFORM_ENERGY) && (p < _parameters->CELL_TRANSFORM_PROB)) {

				//look for neighbor cell
				for (int dx = -2; dx < 3; ++dx) {
					for (int dy = -2; dy < 3; ++dy) {
						if (_cellMap->getCell(_pos + QVector3D(dx, dy, 0.0))) {
							_energyMap->setParticle(_pos, this);
							return false;
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
				return true;
			}
			else {
				_energyMap->setParticle(_pos, this);
				return false;
			}

		}
	}
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
