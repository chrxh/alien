#ifndef SIMULATIONUNITCONTEXTIMPL_H
#define SIMULATIONUNITCONTEXTIMPL_H

#include <QMutex>

#include "model/context/UnitContext.h"
#include "model/context/SpaceMetric.h"

class UnitContextImpl
	: public UnitContext
{
public:
	UnitContextImpl(QObject* parent = nullptr);
	virtual ~UnitContextImpl();

	void init(NumberGenerator* numberGen, SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap
		, MapCompartment* mapCompartment, SymbolTable* symbolTable, SimulationParameters* parameters) override;

	virtual NumberGenerator* getNumberGenerator() const override;
	virtual SpaceMetric* getSpaceMetric () const override;
	virtual CellMap* getCellMap () const override;
	virtual EnergyParticleMap* getEnergyParticleMap () const override;
	virtual MapCompartment* getMapCompartment() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

	virtual uint64_t getTimestamp() const override;
	virtual void incTimestamp() override;

	virtual QList<CellCluster*>& getClustersRef() override;
	virtual QList<EnergyParticle*>& getEnergyParticlesRef () override;

private:
	void deleteClustersAndEnergyParticles();

	QList<CellCluster*> _clusters;
    QList<EnergyParticle*> _energyParticles;
	NumberGenerator* _numberGen = nullptr;
	SpaceMetric* _metric = nullptr;
    CellMap* _cellMap = nullptr;
    EnergyParticleMap* _energyParticleMap = nullptr;
	MapCompartment* _mapCompartment = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;

	uint64_t _timestamp = 0;
};

#endif // SIMULATIONUNITCONTEXTIMPL_H
