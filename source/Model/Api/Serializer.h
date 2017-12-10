#pragma once

#include <QObject>

#include "Definitions.h"

class Serializer
	: public QObject
{
	Q_OBJECT

public:
	Serializer(QObject* parent = nullptr) : QObject(parent) { }
	virtual ~Serializer() = default;

	virtual void serialize(SimulationController* simController, SimulationAccess* access) = 0;
	Q_SIGNAL void serializationFinished();
	virtual string const& retrieveSerializedSimulationContent() = 0;
	virtual string const& retrieveSerializedSimulation() = 0;

	virtual void deserializeSimulationContent(SimulationAccess* access, string const& content) const = 0;
	virtual SimulationController* deserializeSimulation(SimulationAccess* access, string const& content) const = 0;
};
