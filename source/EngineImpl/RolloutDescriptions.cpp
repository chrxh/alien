#include "RolloutDescriptions.h"

#include <boost/range/adaptors.hpp>


namespace
{
    boost::optional<std::list<ConnectionRolloutDescription>> convert(
        std::vector<ConnectionDescription> const& connections)
	{
        std::list<ConnectionRolloutDescription> result;
        for (auto const& connection : connections) {
            ConnectionRolloutDescription connectionChange;
            connectionChange.cellId = connection.cellId;
            connectionChange.distance = connection.distance;
            connectionChange.angleFromPrevious = connection.angleFromPrevious;
            result.emplace_back(connectionChange);
        }
        return result;
    }
}

CellRolloutDescription::CellRolloutDescription(CellDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
    vel = desc.vel;
    energy = desc.energy;
	maxConnections = desc.maxConnections;
    connectingCells = convert(desc.connections);
	tokenBlocked = desc.tokenBlocked;
	tokenBranchNumber = desc.tokenBranchNumber;
	metadata = desc.metadata;
	cellFeatures = desc.cellFeature;
	tokens = desc.tokens;
    tokenUsages = desc.tokenUsages;
}

CellRolloutDescription::CellRolloutDescription(CellDescription const & before, CellDescription const & after)
{
	id = after.id;
	pos = ValueTracker<RealVector2D>(before.pos, after.pos);
    vel = ValueTracker<RealVector2D>(before.vel, after.vel);
    energy = ValueTracker<double>(before.energy, after.energy);
	maxConnections = ValueTracker<int>(before.maxConnections, after.maxConnections);
    connectingCells =
        ValueTracker<std::list<ConnectionRolloutDescription>>(convert(before.connections), convert(after.connections));
    tokenBlocked = ValueTracker<bool>(before.tokenBlocked, after.tokenBlocked);
	tokenBranchNumber = ValueTracker<int>(before.tokenBranchNumber, after.tokenBranchNumber);
	metadata = ValueTracker<CellMetadata>(before.metadata, after.metadata);
	cellFeatures = ValueTracker<CellFeatureDescription>(before.cellFeature, after.cellFeature);
	tokens = ValueTracker<vector<TokenDescription>>(before.tokens, after.tokens);
    tokenUsages = ValueTracker<int>(before.tokenUsages, after.tokenUsages);
}

bool CellRolloutDescription::isEmpty() const
{
    return !pos && !energy && !maxConnections && !connectingCells && !tokenBlocked && !tokenBranchNumber && !metadata
        && !cellFeatures && !tokens;
}

ParticleRolloutDescription::ParticleRolloutDescription(ParticleDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	vel = desc.vel;
	energy = desc.energy;
	metadata = desc.metadata;
}

ParticleRolloutDescription::ParticleRolloutDescription(ParticleDescription const & before, ParticleDescription const & after)
{
	id = after.id;
	pos = ValueTracker<RealVector2D>(before.pos, after.pos);
	vel = ValueTracker<RealVector2D>(before.vel, after.vel);
	energy = ValueTracker<double>(before.energy, after.energy);
	metadata = ValueTracker<ParticleMetadata>(before.metadata, after.metadata);
}

bool ParticleRolloutDescription::isEmpty() const
{
    return !pos && !vel && !energy && !metadata;
}

DataRolloutDescription::DataRolloutDescription(DataDescription const & desc)
{
    for (auto const& cluster : desc.clusters) {
        for (auto const& cell : cluster.cells) {
            addNewCell(cell);
        }
    }
    for (auto const& particle : desc.particles) {
        addNewParticle(particle);
    }
}

DataRolloutDescription::DataRolloutDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
    std::vector<CellDescription> cellsBefore;
    std::vector<CellDescription> cellsAfter;
    for (auto const& cluster : dataBefore.clusters) {
        cellsBefore.insert(cellsBefore.begin(), cluster.cells.begin(), cluster.cells.end());
    }
    for (auto const& cluster : dataAfter.clusters) {
        cellsAfter.insert(cellsAfter.begin(), cluster.cells.begin(), cluster.cells.end());
    }

    unordered_map<uint64_t, int> cellsAfterIndicesByIds;
    for (int index = 0; index < cellsAfter.size(); ++index) {
        cellsAfterIndicesByIds.insert_or_assign(cellsAfter.at(index).id, index);
	}

	for (auto const& cellBefore : cellsBefore) {
        auto cellIdAfterIt = cellsAfterIndicesByIds.find(cellBefore.id);
        if (cellIdAfterIt == cellsAfterIndicesByIds.end()) {
            addDeletedCell(CellRolloutDescription().setId(cellBefore.id).setPos(cellBefore.pos));
		}
		else {
			int cellAfterIndex = cellIdAfterIt->second;
            auto const& cellAfter = cellsAfter.at(cellAfterIndex);
			CellRolloutDescription change(cellBefore, cellAfter);
			if (!change.isEmpty()) {
				addModifiedCell(change);
			}
			cellsAfterIndicesByIds.erase(cellBefore.id);
		}
	}

	for (auto const& cellAfterIndex : cellsAfterIndicesByIds | boost::adaptors::map_values) {
        auto const& cellAfter = cellsAfter.at(cellAfterIndex);
        CellRolloutDescription change(cellAfter);
        addNewCell(change);
	}

	unordered_map<uint64_t, int> particleAfterIndicesByIds;
    for (int index = 0; index < dataAfter.particles.size(); ++index) {
        particleAfterIndicesByIds.insert_or_assign(dataAfter.particles.at(index).id, index);
    }

    for (auto const& particleBefore : dataBefore.particles) {
        auto particleIdAfterIt = particleAfterIndicesByIds.find(particleBefore.id);
        if (particleIdAfterIt == particleAfterIndicesByIds.end()) {
            addDeletedParticle(ParticleRolloutDescription().setId(particleBefore.id).setPos(particleBefore.pos));
        } else {
            int particleAfterIndex = particleIdAfterIt->second;
            auto const& particleAfter = dataAfter.particles.at(particleAfterIndex);
            ParticleRolloutDescription change(particleBefore, particleAfter);
            if (!change.isEmpty()) {
                addModifiedParticle(change);
            }
            particleAfterIndicesByIds.erase(particleBefore.id);
        }
    }

    for (auto const& particleAfterIndexById : particleAfterIndicesByIds) {
        auto const& particleAfter = dataAfter.particles.at(particleAfterIndexById.second);
        addNewParticle(ParticleRolloutDescription(particleAfter));
    }
}

