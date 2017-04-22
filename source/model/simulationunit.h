#ifndef SIMULATIONUNIT_H
#define SIMULATIONUNIT_H

#include <QThread>
#include "definitions.h"

class SimulationUnit : public QObject
{
    Q_OBJECT
public:
    SimulationUnit (SimulationContext* context, QObject* parent = 0);
    ~SimulationUnit ();

public slots:
	void init(uint seed);
	void setContext(SimulationContext* context);

public:
    qreal calcTransEnergy ();
    qreal calcRotEnergy ();
    qreal calcInternalEnergy ();

signals:
    void nextTimestepCalculated ();

public slots:
	void calcNextTimestep();

private:
	void processingEnergyParticles();
	void processingClusterCompletion();
	void processingClusterToken();
	void processingClusterMovement();
	void processingClusterMutationByChance();
	void processingClusterDissipation();
	void processingClusterInit();
	
	void debugCluster(CellCluster* c, int s);
    
	SimulationContext* _context = nullptr;
};

#endif // SIMULATIONUNIT_H
