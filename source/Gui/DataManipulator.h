#pragma once

#include "Model/Entities/Descriptions.h"

#include "Gui/Definitions.h"

class DataManipulator
	: public QObject
{
	Q_OBJECT
public:
	DataManipulator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~DataManipulator() = default;

	virtual void init(SimulationAccess* access, CellConnector* connector);

	virtual DataDescription& getDataRef();
	virtual CellDescription& getCellDescRef(uint64_t cellId);
	virtual ClusterDescription& getClusterDescRef(uint64_t clusterId);

	virtual void setSelection(list<uint64_t> const &cellIds, list<uint64_t> const &particleIds);
	virtual void moveSelection(QVector2D const &delta);
	virtual void moveExtendedSelection(QVector2D const &delta);

	virtual bool isInSelection(list<uint64_t> const &ids) const;
	virtual bool isInSelection(uint64_t id) const; //id can mean cell or particle id
	virtual bool isInExtendedSelection(uint64_t id) const;
	virtual bool areEntitiesSelected() const;
	virtual list<uint64_t> getSelectedCellIds() const;
	virtual list<uint64_t> getSelectedParticleIds() const;

	virtual void dataUpdateRequired(IntRect const& rect) const;

	Q_SIGNAL void notify(set<UpdateTarget> const& targets);

private:
	Q_SLOT void dataFromSimulationAvailable();
	Q_SLOT void sendDataChangesToSimulation(set<UpdateTarget> const& targets);

	void updateAfterCellReconnections();
	void updateInternals(DataDescription const &data);
	ParticleDescription & getParticleDescRef(uint64_t particleId);
	bool isCellPresent(uint64_t cellId);
	bool isParticlePresent(uint64_t particleId);

	SimulationAccess* _access = nullptr;
	CellConnector* _connector = nullptr;
	DataDescription _data;
	DataDescription _unchangedData;

	set<uint64_t> _selectedCellIds;
	set<uint64_t> _selectedClusterIds;
	set<uint64_t> _selectedParticleIds;

	DescriptionNavigationMaps _navi;
};
