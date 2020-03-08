#pragma once

#include "DescriptionHelper.h"
#include "ModelBasic/Physics.h"

class DescriptionHelperImpl
	: public DescriptionHelper
{
	Q_OBJECT
public:
	DescriptionHelperImpl(QObject *parent = nullptr) : DescriptionHelper(parent) { }
	virtual ~DescriptionHelperImpl() = default;

	virtual void init(SimulationContext* context) override;

	virtual void reconnect(DataDescription& data, DataDescription& orgData, unordered_set<uint64_t> const& idsOfChangedCells) override;
	virtual void recluster(DataDescription& data, unordered_set<uint64_t> const& idsOfChangedClusters) override;
    virtual void makeValid(DataDescription& data) override;
    virtual void makeValid(ClusterDescription& cluster) override;
	virtual void makeValid(ParticleDescription& particle) override;

    virtual void duplicate(DataDescription& data, IntVector2D const& origSize, IntVector2D const& size) override;

private:
	list<uint64_t> filterPresentCellIds(unordered_set<uint64_t> const& cellIds) const;
	void updateInternals();
	void updateConnectingCells(list<uint64_t> const &changedCellIds);
	void reclustering(unordered_set<uint64_t> const& clusterIds);

	CellDescription& getCellDescRef(uint64_t cellId);
	void removeConnections(CellDescription &cellDesc);
	void establishNewConnectionsWithNeighborCells(CellDescription &cellDesc);
	void establishNewConnection(CellDescription &cell1, CellDescription &cell2) const;
	double getDistance(CellDescription &cell1, CellDescription &cell2) const;

	list<uint64_t> getCellIdsAtPos(IntVector2D const &pos);

	unordered_set<int> reclusteringSingleClusterAndReturnDiscardedClusterIndices(int clusterIndex, vector<ClusterDescription> &newClusters);
	void lookUpCell(uint64_t cellId, ClusterDescription &newCluster, unordered_set<uint64_t> &lookedUpCellIds, unordered_set<uint64_t> &remainingCellIds);

	void setClusterAttributes(ClusterDescription& cluster);
	double calcAngleBasedOnOrigClusters(vector<CellDescription> const & cells) const;
	Physics::Velocities calcVelocitiesBasedOnOrigClusters(vector<CellDescription> const & cells) const;
	optional<ClusterMetadata> calcMetadataBasedOnOrigClusters(vector<CellDescription> const & cells) const;

	SpaceProperties* _metric = nullptr;
	SimulationParameters _parameters;
	NumberGenerator* _numberGen = nullptr;

	DataDescription* _data = nullptr;
	DataDescription* _origData = nullptr;
	DescriptionNavigator _navi;
	DescriptionNavigator _origNavi;
	unordered_map<int, unordered_map<int, list<uint64_t>>> _cellMap;
};
