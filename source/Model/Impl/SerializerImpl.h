#pragma once

#include <QObject>

#include "Model/Api/Serializer.h"
#include "Definitions.h"

class SerializerImpl
	: public Serializer
{
	Q_OBJECT

public:
	SerializerImpl(QObject *parent = nullptr);
	virtual ~SerializerImpl() = default;

	virtual void serialize(SimulationController* simController, SimulationAccess* access) override;
	virtual string const& retrieveSerializedSimulationContent() override;
	virtual string const& retrieveSerializedSimulation() override;

	virtual void deserializeSimulationContent(SimulationAccess* access, string const& content) const override;
	virtual SimulationController* deserializeSimulation(SimulationAccess* access, string const& content) const override;

private:
	void dataReadyToRetrieve();

	SimulationController* _simController = nullptr;
	SimulationAccess* _access = nullptr;

	string _universeContent;
	string _simulation;
};
