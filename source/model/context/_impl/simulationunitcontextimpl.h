#ifndef SIMULATIONUNITCONTEXTIMPL_H
#define SIMULATIONUNITCONTEXTIMPL_H

#include <QMutex>

#include "model/context/simulationunitcontext.h"
#include "model/context/spacemetric.h"

class SimulationUnitContextImpl
	: public SimulationUnitContext
{
public:
	SimulationUnitContextImpl(QObject* parent = nullptr);
	virtual ~SimulationUnitContextImpl();

	void init(SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap, MapCompartment* mapCompartment, SymbolTable* symbolTable
		, SimulationParameters* parameters) override;

	virtual void lock();
	virtual void unlock();

	virtual SpaceMetric* getTopology () const override;
	virtual CellMap* getCellMap () const override;
	virtual EnergyParticleMap* getEnergyParticleMap () const override;
	virtual MapCompartment* getMapCompartment() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

	virtual QList<CellCluster*>& getClustersRef() override;
	virtual QList<EnergyParticle*>& getEnergyParticlesRef () override;

private:
	void deleteClustersAndEnergyParticles();

	QMutex _mutex;
	
	QList<CellCluster*> _clusters;
    QList<EnergyParticle*> _energyParticles;
    SpaceMetric* _metric = nullptr;
    CellMap* _cellMap = nullptr;
    EnergyParticleMap* _energyParticleMap = nullptr;
	MapCompartment* _mapCompartment = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONUNITCONTEXTIMPL_H
