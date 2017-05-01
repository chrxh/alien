#ifndef SIMULATIONUNITIMPL_H
#define SIMULATIONUNITIMPL_H

#include "model/context/simulationunit.h"

class SimulationUnitImpl
	: public SimulationUnit
{
	Q_OBJECT
public:
	SimulationUnitImpl(QObject* parent = nullptr);
	virtual ~SimulationUnitImpl() {}

public slots:
	virtual void init(SimulationUnitContext* context) override;

public:
	virtual qreal calcTransEnergy() const override;
	virtual qreal calcRotEnergy() const override;
	virtual qreal calcInternalEnergy() const override;

public slots:
	virtual void calcNextTimestep() override;

private:
	void processingEnergyParticles();
	void processingClusterCompletion();
	void processingClusterToken();
	void processingClusterMovement();
	void processingClusterMutationByChance();
	void processingClusterDissipation();
	void processingClusterInit();

	SimulationUnitContext* _context = nullptr;
};

#endif // SIMULATIONUNITIMPL_H
