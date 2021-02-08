#pragma once

#include "ChangeDescriptions.h"

class ENGINEINTERFACE_EXPORT DescriptionHelper
	: public QObject
{
public:
	DescriptionHelper(QObject* parent = nullptr);
	virtual ~DescriptionHelper() = default;

	virtual void init(SimulationContext* context) = 0;

	virtual void reconnect(DataDescription& data, DataDescription& orgData, unordered_set<uint64_t> const& idsOfChangedCells) = 0;
	virtual void recluster(DataDescription& data, unordered_set<uint64_t> const& idsOfChangedClusters) = 0;
    virtual void makeValid(DataDescription& data) = 0;
    virtual void makeValid(ClusterDescription& cluster) = 0;
	virtual void makeValid(ParticleDescription& particle) = 0;

    virtual void duplicate(DataDescription& data, IntVector2D const& origSize, IntVector2D const& size) = 0;
};

