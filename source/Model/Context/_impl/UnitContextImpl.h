#pragma once

#include <QMutex>

#include "Model/Context/UnitContext.h"
#include "Model/Context/SpaceMetric.h"

class UnitContextImpl
	: public UnitContext
{
public:
	UnitContextImpl(QObject* parent = nullptr);
	virtual ~UnitContextImpl();

	void init(NumberGenerator* numberGen, SpaceMetric* metric, CellMap* cellMap, ParticleMap* energyMap
		, MapCompartment* mapCompartment, SymbolTable* symbolTable, SimulationParameters* parameters) override;

	virtual NumberGenerator* getNumberGenerator() const override;
	virtual SpaceMetric* getSpaceMetric () const override;
	virtual CellMap* getCellMap () const override;
	virtual ParticleMap* getEnergyParticleMap () const override;
	virtual MapCompartment* getMapCompartment() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

	virtual uint64_t getTimestamp() const override;
	virtual void incTimestamp() override;

	virtual QList<Cluster*>& getClustersRef() override;
	virtual QList<Particle*>& getParticlesRef () override;

private:
	void deleteClustersAndEnergyParticles();

	QList<Cluster*> _clusters;
    QList<Particle*> _energyParticles;
	NumberGenerator* _numberGen = nullptr;
	SpaceMetric* _metric = nullptr;
    CellMap* _cellMap = nullptr;
    ParticleMap* _energyParticleMap = nullptr;
	MapCompartment* _mapCompartment = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;

	uint64_t _timestamp = 0;
};

