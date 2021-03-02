#pragma once

#include <QObject>

#include "Definitions.h"

class ENGINEINTERFACE_EXPORT Serializer
	: public QObject
{
	Q_OBJECT

public:
	Serializer(QObject* parent = nullptr) : QObject(parent) { }
	virtual ~Serializer() = default;

	virtual void init(SimulationControllerBuildFunc const& controllerBuilder, SimulationAccessBuildFunc const& accessBuilder) = 0;

	struct Settings {
		IntVector2D universeSize;
		map<string, int> typeSpecificData;
        bool duplicateContent;
	};
	virtual void serialize(SimulationController* simController, int typeId, boost::optional<Settings> newSettings = boost::none) = 0;
	Q_SIGNAL void serializationFinished();
    virtual SerializedSimulation const& retrieveSerializedSimulation() = 0;
    virtual SimulationController* deserializeSimulation(std::string const& data) = 0;
    virtual SimulationController* deserializeSimulation(SerializedSimulation const& data) = 0;

	virtual string serializeDataDescription(DataDescription const& desc) const = 0;
	virtual DataDescription deserializeDataDescription(string const& data) = 0;

	virtual string serializeSymbolTable(SymbolTable const* symbolTable) const = 0;
	virtual SymbolTable* deserializeSymbolTable(string const& data) = 0;

	virtual string serializeSimulationParameters(SimulationParameters const& parameters) const = 0;
	virtual SimulationParameters deserializeSimulationParameters(string const& data) = 0;
};
