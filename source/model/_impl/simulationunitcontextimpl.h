#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include <QMutex>

#include "model/simulationunitcontext.h"
#include "model/topology.h"

class SimulationUnitContextImpl : public SimulationUnitContext
{
public:
	SimulationUnitContextImpl();
	SimulationUnitContextImpl(SymbolTable* symbolTable);
	virtual ~SimulationUnitContextImpl();

	void init(Topology* topology) override;

    void lock () override;
    void unlock () override;

    Topology* getTopology () const override;
    EnergyParticleMap* getEnergyParticleMap () const override;
    CellMap* getCellMap () const override;
	SymbolTable* getSymbolTable() const override;
	SimulationParameters* getSimulationParameters() const override;

	QList<CellCluster*>& getClustersRef() override;
    QList<EnergyParticle*>& getEnergyParticlesRef () override;
	std::set<quint64> SimulationUnitContextImpl::getAllCellIds() const override;

private:
	void deleteAll ();
	void deleteClustersAndEnergyParticles();

    QMutex _mutex;
    QList<CellCluster*> _clusters;
    QList<EnergyParticle*> _energyParticles;
    Topology* _topology = nullptr;
    EnergyParticleMap* _energyParticleMap = nullptr;
    CellMap* _cellMap = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters= nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
