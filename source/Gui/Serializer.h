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
	virtual QByteArray const& retrieveSerializedSimulationContent() const;
	virtual QByteArray const& retrieveSerializedSimulation() const;

	virtual void deserializeSimulationContent(SimulationController* simController, QByteArray const& content) const;
	virtual SimulationController* deserializeSimulation(QByteArray const&content) const;

private:
	QByteArray _simulationContent;
	QByteArray _simulation;
};
