#include <qmath.h>

#include "global/ServiceLocator.h"
#include "global/NumberGenerator.h"

#include "model/physics/Physics.h"
#include "model/ModelSettings.h"
#include "model/context/UnitContext.h"
#include "model/context/EnergyParticleMap.h"
#include "model/context/CellMap.h"
#include "model/context/SpaceMetric.h"
#include "model/context/SimulationParameters.h"

#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EntityFactory.h"
#include "EnergyParticleImpl.h"


EnergyParticleImpl::EnergyParticleImpl(UnitContext* context)
	: _context(context)
{
	_id = _context->getNumberGenerator()->getTag();
}

EnergyParticleImpl::EnergyParticleImpl(qreal energy, QVector2D pos, QVector2D vel, UnitContext* context)
	: EnergyParticleImpl(context)
{
	_energy = energy;
	_pos = pos;
	_vel = vel;
}

void EnergyParticleImpl::init(UnitContext * context)
{
	_context = context;
}

//return: false = energy is zero
//        cluster is nonzero if particle transforms into cell
bool EnergyParticleImpl::processingMovement(CellCluster*& cluster)
{
	auto cellMap = _context->getCellMap();
	auto energyMap = _context->getEnergyParticleMap();
	auto parameters = _context->getSimulationParameters();
	energyMap->removeParticleIfPresent(_pos, this);
	move();

	if (EnergyParticle* otherEnergy = energyMap->getParticle(_pos)) {
		collisionWithEnergyParticle(otherEnergy);
		return false;
	}

	//is there a cell at new position?
	Cell* cell(cellMap->getCell(_pos));
	if (cell) {
		collisionWithCell(cell);
		return false;
	}

	//enough energy for cell transformation?
	qreal p(_context->getNumberGenerator()->getRandomReal());
	qreal eKin = Physics::kineticEnergy(1, _vel, 0, 0);
	qreal eNew = _energy - (eKin / parameters->cellMass_Reciprocal);
	if ((eNew >= parameters->cellMinEnergy) && (p < parameters->cellTransformationProb)) {

		//look for neighbor cell
		for (int dx = -2; dx < 3; ++dx) {
			for (int dy = -2; dy < 3; ++dy) {
				if (cellMap->getCell(_pos + QVector2D(dx, dy))) {
					energyMap->setParticle(_pos, this);
					return true;
				}
			}
		}

		//create cell and cluster
		QList< Cell* > cells;
		EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
		CellMetadata meta;
		meta.color = _metadata.color;
		auto desc = CellClusterDescription().setPos(QVector2D(_pos.x(), _pos.y())).setVel(QVector2D(_vel.x(), _vel.y()))
			.addCell(getRandomCellDesciption(eNew).setMetadata(meta));
		cluster = factory->build(desc, _context);
		_energy = 0;
		cluster->drawCellsToMap();
		return false;
	}
	else {
		energyMap->setParticle(_pos, this);
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

CellDescription EnergyParticleImpl::getRandomCellDesciption(double energy) const
{
	auto parameters = _context->getSimulationParameters();
	int randomMaxConnections = _context->getNumberGenerator()->getRandomInt(parameters->cellMaxBonds + 1);
	int randomTokenAccessNumber = _context->getNumberGenerator()->getRandomInt(parameters->cellMaxTokenBranchNumber);
	QByteArray randomData(256, 0);
	for (int i = 0; i < 256; ++i) {
		randomData[i] = _context->getNumberGenerator()->getRandomInt(256);
	}
	Enums::CellFunction::Type randomCellFunction = static_cast<Enums::CellFunction::Type>(_context->getNumberGenerator()->getRandomInt(Enums::CellFunction::_COUNTER));
	return CellDescription().setEnergy(energy).setCellFunction(CellFunctionDescription().setType(randomCellFunction).setData(randomData))
		.setMaxConnections(randomMaxConnections).setTokenAccessNumber(randomTokenAccessNumber);
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
	_context->getSpaceMetric()->correctPosition(_pos);
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

QVector2D EnergyParticleImpl::getPosition() const
{
	return _pos;
}

void EnergyParticleImpl::setPosition(QVector2D value)
{
	_pos = value;
}

QVector2D EnergyParticleImpl::getVelocity() const
{
	return _vel;
}

void EnergyParticleImpl::setVelocity(QVector2D value)
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
