#pragma once

#include "Model/Api/ChangeDescriptions.h"

class MODEL_EXPORT DescriptionHelper
	: public QObject
{
public:
	DescriptionHelper(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~DescriptionHelper() = default;

	virtual void init(SimulationContext* context) = 0;

	virtual void reconnect(DataDescription& data, unordered_set<uint64_t> const& changedCellIds) = 0;
	virtual void recluster(DataDescription& data, unordered_set<uint64_t> const& changedClusterIds) = 0;
	virtual void makeValid(ClusterDescription& cluster) = 0;
	virtual void makeValid(ParticleDescription& particle) = 0;
};

