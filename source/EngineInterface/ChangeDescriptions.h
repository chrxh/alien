#pragma once

#include "Base/Tracker.h"

#include "Descriptions.h"

struct ConnectionChangeDescription
{
    uint64_t cellId;
    float distance;
    float angleFromPrevious;
    bool operator==(ConnectionChangeDescription const& other) const
    {
        if (cellId != other.cellId) {
            return false;
        }
        if (distance != other.distance) {
            return false;
        }
        if (angleFromPrevious != other.angleFromPrevious) {
            return false;
        }
		return true;
	}
    bool operator!=(ConnectionChangeDescription const& other) const { return !(*this == other); }
};

struct CellChangeDescription
{
	uint64_t id = 0;

	ValueTracker<RealVector2D> pos;
    ValueTracker<RealVector2D> vel;
    ValueTracker<double> energy;
	ValueTracker<int> maxConnections;
    ValueTracker<std::list<ConnectionChangeDescription>> connectingCells;
	ValueTracker<bool> tokenBlocked;
	ValueTracker<int> tokenBranchNumber;
	ValueTracker<CellMetadata> metadata;
	ValueTracker<CellFeatureDescription> cellFeatures;
	ValueTracker<vector<TokenDescription>> tokens;
    ValueTracker<int> tokenUsages;

	ENGINEINTERFACE_EXPORT CellChangeDescription() = default;
    ENGINEINTERFACE_EXPORT CellChangeDescription(CellDescription const& desc);
    ENGINEINTERFACE_EXPORT CellChangeDescription(CellDescription const& before, CellDescription const& after);

	ENGINEINTERFACE_EXPORT bool isEmpty() const;
	CellChangeDescription& setId(uint64_t value) { id = value; return *this; }
	CellChangeDescription& setPos(RealVector2D const& value) { pos = value; return *this; }
	CellChangeDescription& setEnergy(double value) { energy = value; return *this; }
	CellChangeDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
    CellChangeDescription& setConnectingCells(list<ConnectionChangeDescription> const& value)
    {
        connectingCells = value;
        return *this;
    }
	CellChangeDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellChangeDescription& setTokenBranchNumber(int value) { tokenBranchNumber = value; return *this; }
	CellChangeDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellChangeDescription& setCellFunction(CellFeatureDescription const& value) { cellFeatures = value; return *this; }
    CellChangeDescription& setTokenUsages(int value) { tokenUsages = value; return *this; }
};

struct ParticleChangeDescription
{
	uint64_t id = 0;

	ValueTracker<RealVector2D> pos;
	ValueTracker<RealVector2D> vel;
	ValueTracker<double> energy;
	ValueTracker<ParticleMetadata> metadata;

	ENGINEINTERFACE_EXPORT ParticleChangeDescription() = default;
    ENGINEINTERFACE_EXPORT ParticleChangeDescription(ParticleDescription const& desc);
    ENGINEINTERFACE_EXPORT ParticleChangeDescription(
        ParticleDescription const& before,
        ParticleDescription const& after);

	ENGINEINTERFACE_EXPORT bool isEmpty() const;
	ParticleChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleChangeDescription& setPos(RealVector2D const& value) { pos = value; return *this; }
	ParticleChangeDescription& setVel(RealVector2D const& value) { vel = value; return *this; }
	ParticleChangeDescription& setEnergy(double value) { energy = value; return *this; }
};

struct DataChangeDescription
{
    vector<StateTracker<CellChangeDescription>> cells;
    vector<StateTracker<ParticleChangeDescription>> particles;

	ENGINEINTERFACE_EXPORT DataChangeDescription() = default;
    ENGINEINTERFACE_EXPORT DataChangeDescription(DataDescription const& desc);
    ENGINEINTERFACE_EXPORT DataChangeDescription(DataDescription const& dataBefore, DataDescription const& dataAfter);

	DataChangeDescription& addNewCell(CellChangeDescription const& value)
	{
		cells.emplace_back(StateTracker<CellChangeDescription>(value, StateTracker<CellChangeDescription>::State::Added));
		return *this;
	}
	DataChangeDescription& addModifiedCell(CellChangeDescription const& value)
	{
        cells.emplace_back(
            StateTracker<CellChangeDescription>(value, StateTracker<CellChangeDescription>::State::Modified));
		return *this;
	}
    DataChangeDescription& addModifiedCell(list<CellChangeDescription> const& value)
	{
		for (auto const &cell : value) {
			addModifiedCell(cell);
		}
		return *this;
	}
    DataChangeDescription& addDeletedCell(CellChangeDescription const& value)
	{
        cells.emplace_back(
            StateTracker<CellChangeDescription>(value, StateTracker<CellChangeDescription>::State::Deleted));
		return *this;
	}
	DataChangeDescription& addNewParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleChangeDescription>(value, StateTracker<ParticleChangeDescription>::State::Added));
		return *this;
	}
	DataChangeDescription& addModifiedParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleChangeDescription>(value, StateTracker<ParticleChangeDescription>::State::Modified));
		return *this;
	}
	DataChangeDescription& addDeletedParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleChangeDescription>(value, StateTracker<ParticleChangeDescription>::State::Deleted));
		return *this;
	}
	void clear()
	{
		cells.clear();
		particles.clear();
	}
	bool empty()
	{
		return cells.empty() && particles.empty();
	}
};


