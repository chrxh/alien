#pragma once

#include <QObject>

#include "Definitions.h"

class MODEL_EXPORT Serializer
	: public QObject
{
	Q_OBJECT

public:
	Serializer(QObject* parent = nullptr) : QObject(parent) { }
	virtual ~Serializer() = default;

	virtual void init() = 0;

	virtual void serialize(SimulationController* simController) = 0;
	Q_SIGNAL void serializationFinished();
	virtual string const& retrieveSerializedSimulationContent() = 0;
	virtual string const& retrieveSerializedSimulation() = 0;
	virtual void deserializeSimulationContent(string const& content) const = 0;
	virtual SimulationController* deserializeSimulation(string const& content) = 0;

	virtual string serializeSymbolTable(SymbolTable* symbolTable) const = 0;
	virtual SymbolTable* deserializeSymbolTable(string const& data) = 0;

	virtual string serializeSimulationParameters(SimulationParameters* parameters) const = 0;
	virtual SimulationParameters* deserializeSimulationParameters(string const& data) = 0;
};
