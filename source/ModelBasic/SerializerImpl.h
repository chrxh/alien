#pragma once

#include <QObject>

#include "ModelBasic/Serializer.h"
#include "Definitions.h"

class SerializerImpl
	: public Serializer
{
	Q_OBJECT

public:
	SerializerImpl(QObject *parent = nullptr);
	virtual ~SerializerImpl() = default;

	virtual void init(SimulationControllerBuildFunc const& controllerBuilder, SimulationAccessBuildFunc const& accessBuilder) override;

	virtual void serialize(SimulationController* simController, int typeId, optional<Settings> newSettings = boost::none) override;
	virtual string const& retrieveSerializedSimulation() override;
	virtual SimulationController* deserializeSimulation(string const& content) override;

	virtual string serializeDataDescription(DataDescription const& desc) const override;
	virtual DataDescription deserializeDataDescription(string const& data) override;

	virtual string serializeSymbolTable(SymbolTable const* symbolTable) const override;
	virtual SymbolTable* deserializeSymbolTable(string const& data) override;

	virtual string serializeSimulationParameters(SimulationParameters const& parameters) const override;
	virtual SimulationParameters deserializeSimulationParameters(string const& data) override;

private:
	Q_SLOT void dataReadyToRetrieve();

	void buildAccess(SimulationController* controller);

	SimulationControllerBuildFunc _controllerBuilder;
	SimulationAccessBuildFunc _accessBuilder;
	SimulationAccess* _access = nullptr;

	struct ConfigToSerialize {
		SimulationParameters parameters;
		SymbolTable const* symbolTable;
		IntVector2D universeSize;
		int typeId;
		map<string, int> typeSpecificData;
		uint timestep;
	};
	ConfigToSerialize _configToSerialize;
	string _serializedSimulation;

	list<QMetaObject::Connection> _connections;
};
