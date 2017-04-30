#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include <QList>
#include <QSize>

#include "definitions.h"

class SimulationUnitContext
{
public:
    virtual ~SimulationUnitContext () {}

	virtual void init(Topology* topology) = 0;

    virtual void lock () = 0;
    virtual void unlock () = 0;
	
    virtual Topology* getTopology () const = 0;
	virtual EnergyParticleMap* getEnergyParticleMap() const = 0;
	virtual CellMap* getCellMap() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;

	virtual QList<CellCluster*>& getClustersRef() = 0;
    virtual QList<EnergyParticle*>& getEnergyParticlesRef () = 0;
	virtual std::set<quint64> getAllCellIds() const = 0;
};

#endif // SIMULATIONCONTEXT_H
