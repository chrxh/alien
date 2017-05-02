#ifndef UNITIMPL_H
#define UNITIMPL_H

#include "model/context/unit.h"

class UnitImpl
	: public Unit
{
	Q_OBJECT
public:
	UnitImpl(QObject* parent = nullptr);
	virtual ~UnitImpl() {}

public slots:
	virtual void init(UnitContext* context) override;

public:
	virtual UnitContext* getContext() const override;
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

	UnitContext* _context = nullptr;
};

#endif // UNITIMPL_H
