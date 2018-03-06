#pragma once

#include <QThread>
#include "Model/Api/Definitions.h"

class Unit
	: public QObject
{
    Q_OBJECT
public:
	Unit(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~Unit() = default;

	virtual void init(UnitContext* context) = 0;
	Q_SLOT virtual void calculateTimestep() = 0;
	Q_SIGNAL void timestepCalculated ();

	virtual UnitContext* getContext() const = 0;
	virtual qreal calcTransEnergy() const = 0;
	virtual qreal calcRotEnergy () const = 0;
	virtual qreal calcInternalEnergy() const = 0;


};
