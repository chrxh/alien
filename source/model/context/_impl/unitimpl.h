#ifndef UNITIMPL_H
#define UNITIMPL_H

#include "model/Context/Unit.h"

class UnitImpl
	: public Unit
{
	Q_OBJECT
public:
	UnitImpl(QObject* parent = nullptr);
	virtual ~UnitImpl() {}

	virtual void init(UnitContext* context) override;
	virtual UnitContext* getContext() const override;
	Q_SLOT virtual void calculateTimestep() override;

	virtual qreal calcTransEnergy() const override;
	virtual qreal calcRotEnergy() const override;
	virtual qreal calcInternalEnergy() const override;

private:
	void processingClustersInit();
	void processingClustersDissipation();
	void processingClustersMutationByChance();
	void processingClustersMovement();
	void processingClustersToken();
	void processingClustersCompletion();
	void processingClustersCompartmentAllocation();

	void processingEnergyParticles();
	void processingEnergyParticlesCompartmentAllocation();

	UnitContext* _context = nullptr;
};

#endif // UNITIMPL_H
