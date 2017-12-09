#pragma once

#include <QObject>

#include "Model/Api/Definitions.h"
#include "Definitions.h"

class Serializer : public QObject
{
	Q_OBJECT

public:
	Serializer(QObject *parent);
	virtual ~Serializer() = default;

	virtual void serialize(SimulationController* simController, SimulationAccess* access);
	Q_SIGNAL void serializationFinished();
	virtual string const& retrieveSerializedSimulationContent();
	virtual string const& retrieveSerializedSimulation();

	virtual void deserializeSimulationContent(SimulationController* simController, string const& content) const;
	virtual SimulationController* deserializeSimulation(string const& content) const;

private:
	SimulationAccess* _access = nullptr;

	string _simulationContent;
	string _simulation;
};
