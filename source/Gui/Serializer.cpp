#include "Serializer.h"

Serializer::Serializer(QObject *parent)
	: QObject(parent)
{
}

void Serializer::serialize(SimulationController * simController, SimulationAccess * access)
{
}

QByteArray const & Serializer::retrieveSerializedSimulationContent() const
{
	return _simulationContent;
}

QByteArray const & Serializer::retrieveSerializedSimulation() const
{
	return _simulation;
}

void Serializer::deserializeSimulationContent(SimulationController * simController, QByteArray const & content) const
{
}

SimulationController * Serializer::deserializeSimulation(QByteArray const & content) const
{
	return nullptr;
}
