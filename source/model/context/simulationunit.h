#ifndef SIMULATIONUNIT_H
#define SIMULATIONUNIT_H

#include <QThread>
#include "model/definitions.h"

class SimulationUnit
	: public QObject
{
    Q_OBJECT
public:
    SimulationUnit (QObject* parent = nullptr);
	virtual ~SimulationUnit ();

public slots:
	void init(SimulationUnitContext* context);

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
    
	SimulationUnitContext* _context = nullptr;
};

#endif // SIMULATIONUNIT_H
