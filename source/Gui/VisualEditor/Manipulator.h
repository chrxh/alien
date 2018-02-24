#pragma once

#include <QObject>

#include "Gui/Definitions.h"

class Manipulator : public QObject
{
	Q_OBJECT

public:
	Manipulator(QObject *parent = nullptr);
	virtual ~Manipulator() = default;

	void init(SimulationContext* context);

	void applyForce(QVector2D const& pos, QVector2D const& force);

private:
	Q_SLOT void dataReadyToRetrieve();

	list<QMetaObject::Connection> _connections;

	SimulationAccess* _access = nullptr;

	bool _waitingForData = false;
	QVector2D _applyAtPos;
	QVector2D _applyForce;
};
