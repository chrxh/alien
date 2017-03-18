#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include <QMutex>

#include "model/simulationcontext.h"
#include "model/topology.h"

class SimulationContextImpl : public SimulationContext
{
public:
	SimulationContextImpl();
	SimulationContextImpl(SymbolTable* symbolTable);
	virtual ~SimulationContextImpl();

	void init(IntVector2D size) override;
	void initWithoutTopology() override;

    void lock () override;
    void unlock () override;

    Topology* getTopology () const override;
    EnergyParticleMap* getEnergyParticleMap () const override;
    CellMap* getCellMap () const override;
	SymbolTable* getSymbolTable() const override;
	SimulationParameters* getSimulationParameters() const override;

	QList<CellCluster*>& getClustersRef() override;
    QList<EnergyParticle*>& getEnergyParticlesRef () override;
	std::set<quint64> SimulationContextImpl::getAllCellIds() const override;

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
