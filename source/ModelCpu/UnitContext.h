#pragma once

#include "Definitions.h"

class UnitContext
	: public QObject
{
	Q_OBJECT
public:
	UnitContext(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~UnitContext() = default;
	
	virtual void init(NumberGenerator* numberGen, SpaceProperties* spaceProp, CellMap* cellMap, ParticleMap* energyMap
		, MapCompartment* mapCompartment, SimulationParameters const& parameters) = 0;

	virtual NumberGenerator* getNumberGenerator() const = 0;
    virtual SpaceProperties* getSpaceProperties() const = 0;
	virtual CellMap* getCellMap() const = 0;
	virtual ParticleMap* getParticleMap() const = 0;
	virtual MapCompartment* getMapCompartment() const = 0;
	virtual SimulationParameters const& getSimulationParameters() const = 0;

	virtual uint64_t getTimestamp() const = 0;
	virtual void incTimestamp() = 0;

	virtual void setSimulationParameters(SimulationParameters const& parameters) = 0;
	virtual QList<Cluster*>& getClustersRef() = 0;
	virtual QList<Particle*>& getParticlesRef() = 0;
};
