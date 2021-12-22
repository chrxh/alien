#pragma once

#include "Base/Tracker.h"

#include "EngineInterface/Descriptions.h"

struct ConnectionRolloutDescription
{
    uint64_t cellId;
    float distance;
    float angleFromPrevious;
    bool operator==(ConnectionRolloutDescription const& other) const
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
    bool operator!=(ConnectionRolloutDescription const& other) const { return !(*this == other); }
};

struct CellRolloutDescription
{
	uint64_t id = 0;

	ValueTracker<RealVector2D> pos;
    ValueTracker<RealVector2D> vel;
    ValueTracker<double> energy;
	ValueTracker<int> maxConnections;
    ValueTracker<std::list<ConnectionRolloutDescription>> connectingCells;
	ValueTracker<bool> tokenBlocked;
	ValueTracker<int> tokenBranchNumber;
	ValueTracker<CellMetadata> metadata;
	ValueTracker<CellFeatureDescription> cellFeatures;
	ValueTracker<vector<TokenDescription>> tokens;
    ValueTracker<int> tokenUsages;

	ENGINEINTERFACE_EXPORT CellRolloutDescription() = default;
    ENGINEINTERFACE_EXPORT CellRolloutDescription(CellDescription const& desc);
    ENGINEINTERFACE_EXPORT CellRolloutDescription(CellDescription const& before, CellDescription const& after);

	ENGINEINTERFACE_EXPORT bool isEmpty() const;
	CellRolloutDescription& setId(uint64_t value) { id = value; return *this; }
	CellRolloutDescription& setPos(RealVector2D const& value) { pos = value; return *this; }
	CellRolloutDescription& setEnergy(double value) { energy = value; return *this; }
	CellRolloutDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
    CellRolloutDescription& setConnectingCells(list<ConnectionRolloutDescription> const& value)
    {
        connectingCells = value;
        return *this;
    }
	CellRolloutDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellRolloutDescription& setTokenBranchNumber(int value) { tokenBranchNumber = value; return *this; }
	CellRolloutDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellRolloutDescription& setCellFunction(CellFeatureDescription const& value) { cellFeatures = value; return *this; }
    CellRolloutDescription& setTokenUsages(int value) { tokenUsages = value; return *this; }
};

struct ParticleRolloutDescription
{
	uint64_t id = 0;

	ValueTracker<RealVector2D> pos;
	ValueTracker<RealVector2D> vel;
	ValueTracker<double> energy;
	ValueTracker<ParticleMetadata> metadata;

	ENGINEINTERFACE_EXPORT ParticleRolloutDescription() = default;
    ENGINEINTERFACE_EXPORT ParticleRolloutDescription(ParticleDescription const& desc);
    ENGINEINTERFACE_EXPORT ParticleRolloutDescription(
        ParticleDescription const& before,
        ParticleDescription const& after);

	ENGINEINTERFACE_EXPORT bool isEmpty() const;
	ParticleRolloutDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleRolloutDescription& setPos(RealVector2D const& value) { pos = value; return *this; }
	ParticleRolloutDescription& setVel(RealVector2D const& value) { vel = value; return *this; }
	ParticleRolloutDescription& setEnergy(double value) { energy = value; return *this; }
};

struct DataRolloutDescription
{
    vector<StateTracker<CellRolloutDescription>> cells;
    vector<StateTracker<ParticleRolloutDescription>> particles;

	ENGINEINTERFACE_EXPORT DataRolloutDescription() = default;
    ENGINEINTERFACE_EXPORT DataRolloutDescription(DataDescription const& desc);
    ENGINEINTERFACE_EXPORT DataRolloutDescription(DataDescription const& dataBefore, DataDescription const& dataAfter);

	DataRolloutDescription& addNewCell(CellRolloutDescription const& value)
	{
		cells.emplace_back(StateTracker<CellRolloutDescription>(value, StateTracker<CellRolloutDescription>::State::Added));
		return *this;
	}
	DataRolloutDescription& addModifiedCell(CellRolloutDescription const& value)
	{
        cells.emplace_back(
            StateTracker<CellRolloutDescription>(value, StateTracker<CellRolloutDescription>::State::Modified));
		return *this;
	}
    DataRolloutDescription& addModifiedCell(list<CellRolloutDescription> const& value)
	{
		for (auto const &cell : value) {
			addModifiedCell(cell);
		}
		return *this;
	}
    DataRolloutDescription& addDeletedCell(CellRolloutDescription const& value)
	{
        cells.emplace_back(
            StateTracker<CellRolloutDescription>(value, StateTracker<CellRolloutDescription>::State::Deleted));
		return *this;
	}
	DataRolloutDescription& addNewParticle(ParticleRolloutDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleRolloutDescription>(value, StateTracker<ParticleRolloutDescription>::State::Added));
		return *this;
	}
	DataRolloutDescription& addModifiedParticle(ParticleRolloutDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleRolloutDescription>(value, StateTracker<ParticleRolloutDescription>::State::Modified));
		return *this;
	}
	DataRolloutDescription& addDeletedParticle(ParticleRolloutDescription const& value)
	{
		particles.emplace_back(StateTracker<ParticleRolloutDescription>(value, StateTracker<ParticleRolloutDescription>::State::Deleted));
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


