#pragma once

#include <QObject>

#include "Definitions.h"

class Manipulator : public QObject
{
	Q_OBJECT

public:
	Manipulator(QObject *parent = nullptr);
	virtual ~Manipulator() = default;

	void init(SimulationContext* context, SimulationAccess* access);

	void applyForce(QVector2D const& pos, QVector2D const& deltaPos);
	void applyRotation(QVector2D const& pos, QVector2D const& deltaPos);

private:
	void proceedManipulation(QVector2D const& pos, QVector2D const& deltaPos);
	Q_SLOT void dataReadyToRetrieve();

	list<QMetaObject::Connection> _connections;

	SimulationAccess* _access = nullptr;

	bool _waitingForData = false;
	QVector2D _applyAtPos;
	QVector2D _applyForce;
	enum class Mode { ApplyForce, ApplyRotation };
	Mode _mode = Mode::ApplyForce;
};
