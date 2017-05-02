#ifndef UNIT_H
#define UNIT_H

#include <QThread>
#include "model/definitions.h"

class Unit
	: public QObject
{
    Q_OBJECT
public:
	Unit(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~Unit() {}

public slots:
	virtual void init(UnitContext* context) = 0;

public:
	virtual UnitContext* getContext() const = 0;
	virtual qreal calcTransEnergy() const = 0;
	virtual qreal calcRotEnergy () const = 0;
	virtual qreal calcInternalEnergy() const = 0;

signals:
    void nextTimestepCalculated ();

public slots:
	virtual void calcNextTimestep() = 0;
};

#endif // UNIT_H
