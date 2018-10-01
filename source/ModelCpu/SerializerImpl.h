#pragma once

#include <QObject>

#include "ModelInterface/Serializer.h"
#include "Definitions.h"

class SerializerImpl
	: public Serializer
{
	Q_OBJECT

public:
	SerializerImpl(QObject *parent = nullptr);
	virtual ~SerializerImpl() = default;

	virtual void init() override;

	virtual void serialize(SimulationController* simController, SerializeOptions options = SerializeOptions()) override;
	virtual string const& retrieveSerializedSimulation() override;
	virtual SimulationController* deserializeSimulation(string const& content) override;

	virtual string serializeDataDescription(DataDescription const& desc) const override;
	virtual DataDescription deserializeDataDescription(string const& data) override;

	virtual string serializeSymbolTable(SymbolTable const* symbolTable) const override;
	virtual SymbolTable* deserializeSymbolTable(string const& data) override;

	virtual string serializeSimulationParameters(SimulationParameters const* parameters) const override;
	virtual SimulationParameters* deserializeSimulationParameters(string const& data) override;

private:
	Q_SLOT void dataReadyToRetrieve();

	SimulationAccess* _access = nullptr;

	struct ConfigToSerialize {
		SimulationParameters const* parameters;
		SymbolTable const* symbolTable;
		IntVector2D universeSize;
		IntVector2D gridSize;
		uint maxThreads;
		uint timestep;
	};
	ConfigToSerialize _configToSerialize;
	string _serializedSimulation;
};
