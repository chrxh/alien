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
	virtual void init(SimulationUnitContext* context) = 0;

public:
    virtual qreal calcTransEnergy () const = 0;
	virtual qreal calcRotEnergy () const = 0;
	virtual qreal calcInternalEnergy() const = 0;

signals:
    void nextTimestepCalculated ();

public slots:
	virtual void calcNextTimestep() = 0;
};

#endif // SIMULATIONUNIT_H
