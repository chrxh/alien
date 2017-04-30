#ifndef SIMULATIONUNITCONTEXTIMPL_H
#define SIMULATIONUNITCONTEXTIMPL_H

#include <QMutex>

#include "model/context/simulationunitcontext.h"
#include "model/context/topology.h"

class SimulationUnitContextImpl
	: public SimulationUnitContext
{
public:
	SimulationUnitContextImpl(QObject* parent = nullptr);
	virtual ~SimulationUnitContextImpl();

	void init(Topology* topology, SymbolTable * symbolTable, SimulationParameters* parameters) override;

    Topology* getTopology () const override;
    EnergyParticleMap* getEnergyParticleMap () const override;
    CellMap* getCellMap () const override;
	SymbolTable* getSymbolTable() const override;
	SimulationParameters* getSimulationParameters() const override;

	QList<CellCluster*>& getClustersRef() override;
    QList<EnergyParticle*>& getEnergyParticlesRef () override;

private:
	void deleteClustersAndEnergyParticles();

    QList<CellCluster*> _clusters;
    QList<EnergyParticle*> _energyParticles;
    Topology* _topology = nullptr;
    EnergyParticleMap* _energyParticleMap = nullptr;
    CellMap* _cellMap = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONUNITCONTEXTIMPL_H
