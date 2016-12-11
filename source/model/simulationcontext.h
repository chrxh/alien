#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include "definitions.h"
#include <QList>
#include <QSize>

class SimulationContext
{
public:
    virtual ~SimulationContext () {}

    virtual void lock () = 0;
    virtual void unlock () = 0;
	
    virtual Topology* getTopology () const = 0;
	virtual EnergyParticleMap* getEnergyParticleMap() const = 0;
	virtual CellMap* getCellMap() const = 0;
	virtual QList<CellCluster*>& getClustersRef() = 0;
    virtual QList<EnergyParticle*>& getEnergyParticlesRef () = 0;
};

#endif // SIMULATIONCONTEXT_H
