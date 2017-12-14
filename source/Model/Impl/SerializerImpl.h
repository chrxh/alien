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

	virtual void init(SimulationAccess* access) override;

	virtual void serialize(SimulationController* simController) override;
	virtual string const& retrieveSerializedSimulationContent() override;
	virtual string const& retrieveSerializedSimulation() override;

	virtual void deserializeSimulationContent(string const& content) const override;
	virtual SimulationController* deserializeSimulation(string const& content) const override;

private:
	Q_SLOT void dataReadyToRetrieve();

	SimulationAccess* _access = nullptr;

	bool _serializationInProgress = false;
	struct ConfigToSerialize {
		SimulationParameters* parameters;
		SymbolTable* symbolTable;
		IntVector2D universeSize;
		IntVector2D gridSize;
		int maxThreads;
	};
	ConfigToSerialize _configToSerialize;
	string _serializedSimulationContent;
	string _serializedSimulation;
};
