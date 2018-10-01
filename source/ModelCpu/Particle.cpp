#include <qmath.h>

#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"

#include "ModelInterface/Settings.h"
#include "ModelInterface/Physics.h"
#include "UnitContext.h"
#include "ParticleMap.h"
#include "CellMap.h"
#include "SpacePropertiesImpl.h"
#include "ModelInterface/SimulationParameters.h"

#include "Cell.h"
#include "Cluster.h"
#include "EntityFactory.h"
#include "Particle.h"


Particle::Particle(uint64_t id, qreal energy, QVector2D pos, QVector2D vel, UnitContext* context)
	: EntityWithTimestamp(context)
	, _id(id)
{
	_energy = energy;
	setPosition(pos);
	_vel = vel;
}

ParticleDescription Particle::getDescription(ResolveDescription const& resolveDescription) const
{
	ParticleDescription result;
	result.setPos(_pos).setVel(_vel).setEnergy(_energy);
	if (resolveDescription.resolveIds) {
		result.setId(_id);
	}
	return result;
}

void Particle::applyChangeDescription(ParticleChangeDescription const & change)
{
	if (change.pos) {
		setPosition(*change.pos);
	}
	if (change.vel) {
		_vel = *change.vel;
	}
	if (change.energy) {
		_energy = *change.energy;
	}
	if (change.metadata) {
		_metadata = *change.metadata;
	}
}

//return: false = energy is zero
//        cluster is nonzero if particle transforms into cell
bool Particle::processingMovement(Cluster*& cluster)
{
	if (!isTimestampFitting()) {
		return true;
	}

	auto cellMap = _context->getCellMap();
	auto energyMap = _context->getParticleMap();
	auto parameters = _context->getSimulationParameters();
	energyMap->removeParticleIfPresent(_pos, this);
	move();

	if (Particle* otherEnergy = energyMap->getParticle(_pos)) {
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
		auto desc = ClusterDescription().setPos(_pos).setVel(_vel)
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

void Particle::clearParticleFromMap()
{
	auto energyMap = _context->getParticleMap();
	energyMap->removeParticleIfPresent(getPosition(), this);
}

void Particle::drawParticleToMap()
{
	auto energyMap = _context->getParticleMap();
	energyMap->setParticle(getPosition(), this);
}

void Particle::collisionWithCell(Cell* cell)
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

CellDescription Particle::getRandomCellDesciption(double energy) const
{
	auto parameters = _context->getSimulationParameters();
	int randomMaxConnections = _context->getNumberGenerator()->getRandomInt(parameters->cellMaxBonds + 1);
	int randomTokenAccessNumber = _context->getNumberGenerator()->getRandomInt(parameters->cellMaxTokenBranchNumber);
	QByteArray randomData(255, 0);
	for (int i = 0; i < 255; ++i) {
		randomData[i] = _context->getNumberGenerator()->getRandomInt(256);
	}
	Enums::CellFunction::Type randomCellFunction = static_cast<Enums::CellFunction::Type>(_context->getNumberGenerator()->getRandomInt(Enums::CellFunction::_COUNTER));
	return CellDescription().setEnergy(energy).setCellFeature(CellFeatureDescription().setType(randomCellFunction).setConstData(randomData))
		.setMaxConnections(randomMaxConnections).setTokenBranchNumber(randomTokenAccessNumber);
}

void Particle::collisionWithEnergyParticle(Particle* otherEnergy)
{
	//particle with most energy inherits color
	if (otherEnergy->getEnergy() < _energy)
		otherEnergy->setMetadata(_metadata);

	otherEnergy->setEnergy(otherEnergy->getEnergy() + _energy);
	_energy = 0;
	otherEnergy->setVelocity((otherEnergy->getVelocity() + _vel) / 2.0);
}

void Particle::move()
{
	_pos += _vel;
	_context->getSpaceProperties()->correctPosition(_pos);
}

qreal Particle::getEnergy() const
{
	return _energy;
}

void Particle::setEnergy(qreal value)
{
	_energy = value;
}

QVector2D Particle::getPosition() const
{
	return _pos;
}

void Particle::setPosition(QVector2D value)
{
	_pos = value;
	_context->getSpaceProperties()->correctPosition(_pos);
}

QVector2D Particle::getVelocity() const
{
	return _vel;
}

void Particle::setVelocity(QVector2D value)
{
	_vel = value;
}

quint64 Particle::getId() const
{
	return _id;
}

void Particle::setId(quint64 value)
{
	_id = value;
}

ParticleMetadata Particle::getMetadata() const
{
	return _metadata;
}

void Particle::setMetadata(ParticleMetadata value)
{
	_metadata = value;
}
