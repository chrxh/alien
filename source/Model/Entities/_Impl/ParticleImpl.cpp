#include <qmath.h>

#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"

#include "Model/Physics/Physics.h"
#include "Model/Settings.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/SpaceMetric.h"
#include "Model/Context/SimulationParameters.h"

#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/EntityFactory.h"
#include "ParticleImpl.h"


ParticleImpl::ParticleImpl(uint64_t id, qreal energy, QVector2D pos, QVector2D vel, UnitContext* context)
	: Particle(context)
	, _id(id)
{
	_energy = energy;
	_pos = pos;
	_vel = vel;
}

ParticleDescription ParticleImpl::getDescription() const
{
	ParticleDescription result;
	result.setId(_id).setPos(_pos).setEnergy(_energy);
	return result;
}

void ParticleImpl::applyChangeDescription(ParticleChangeDescription const & change)
{
	if (change.pos) {
		_pos = *change.pos;
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
bool ParticleImpl::processingMovement(Cluster*& cluster)
{
	if (!isTimestampFitting()) {
		return true;
	}

	auto cellMap = _context->getCellMap();
	auto energyMap = _context->getEnergyParticleMap();
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

void ParticleImpl::collisionWithCell(Cell* cell)
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

CellDescription ParticleImpl::getRandomCellDesciption(double energy) const
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
		.setMaxConnections(randomMaxConnections).setTokenBranchNumber(randomTokenAccessNumber);
}

void ParticleImpl::collisionWithEnergyParticle(Particle* otherEnergy)
{
	//particle with most energy inherits color
	if (otherEnergy->getEnergy() < _energy)
		otherEnergy->setMetadata(_metadata);

	otherEnergy->setEnergy(otherEnergy->getEnergy() + _energy);
	_energy = 0;
	otherEnergy->setVelocity((otherEnergy->getVelocity() + _vel) / 2.0);
}

void ParticleImpl::move()
{
	_pos += _vel;
	_context->getSpaceMetric()->correctPosition(_pos);
}

void ParticleImpl::serializePrimitives(QDataStream& stream) const
{
	stream << _energy << _pos << _vel << _id;
}

void ParticleImpl::deserializePrimitives(QDataStream& stream)
{
	stream >> _energy >> _pos >> _vel >> _id;
}

qreal ParticleImpl::getEnergy() const
{
	return _energy;
}

void ParticleImpl::setEnergy(qreal value)
{
	_energy = value;
}

QVector2D ParticleImpl::getPosition() const
{
	return _pos;
}

void ParticleImpl::setPosition(QVector2D value)
{
	_pos = value;
}

QVector2D ParticleImpl::getVelocity() const
{
	return _vel;
}

void ParticleImpl::setVelocity(QVector2D value)
{
	_vel = value;
}

quint64 ParticleImpl::getId() const
{
	return _id;
}

void ParticleImpl::setId(quint64 value)
{
	_id = value;
}

ParticleMetadata ParticleImpl::getMetadata() const
{
	return _metadata;
}

void ParticleImpl::setMetadata(ParticleMetadata value)
{
	_metadata = value;
}
