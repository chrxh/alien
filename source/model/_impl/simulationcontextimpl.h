#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include "model/simulationcontext.h"
#include "model/topology.h"
#include <QMutex>

class SimulationContextImpl : public SimulationContext
{
public:
	SimulationContextImpl();
	SimulationContextImpl(IntVector2D size);
	virtual ~SimulationContextImpl();

    void lock ();
    void unlock ();

    Topology* getTopology () const;
    EnergyParticleMap* getEnergyParticleMap () const;
    CellMap* getCellMap () const;
	QList<CellCluster*>& getClustersRef();
    QList<EnergyParticle*>& getEnergyParticlesRef ();

private:
	void deleteAttributes ();

    QMutex _mutex;
    QList<CellCluster*> _clusters;
    QList<EnergyParticle*> _energyParticles;
    Topology* _topology = nullptr;
    EnergyParticleMap* _energyParticleMap = nullptr;
    CellMap* _cellMap = nullptr;

};

#endif // SIMULATIONCONTEXTIMPL_H
