#include "ChangeDescriptions.h"

#include <boost/range/adaptors.hpp>


namespace
{
    boost::optional<std::list<ConnectionChangeDescription>> convert(
        boost::optional<list<uint64_t>> const& connectingCellIds)
	{
        if (!connectingCellIds) {
            return boost::none;
        }
        std::list<ConnectionChangeDescription> result;
        for (auto const& connectingCellId : *connectingCellIds) {
            ConnectionChangeDescription connection;
            connection.cellId = connectingCellId;
            result.emplace_back(connection);
        }
        return result;
    }
}

CellChangeDescription::CellChangeDescription(CellDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	energy = desc.energy;
	maxConnections = desc.maxConnections;
    connectingCells = convert(desc.connectingCells);
	tokenBlocked = desc.tokenBlocked;
	tokenBranchNumber = desc.tokenBranchNumber;
	metadata = desc.metadata;
	cellFeatures = desc.cellFeature;
	tokens = desc.tokens;
    tokenUsages = desc.tokenUsages;
}

CellChangeDescription::CellChangeDescription(CellDescription const & before, CellDescription const & after)
{
	id = after.id;
	pos = ValueTracker<QVector2D>(before.pos, after.pos);
	energy = ValueTracker<double>(before.energy, after.energy);
	maxConnections = ValueTracker<int>(before.maxConnections, after.maxConnections);
    connectingCells = ValueTracker<list<ConnectionChangeDescription>>(
        convert(before.connectingCells), convert(after.connectingCells));
	tokenBlocked = ValueTracker<bool>(before.tokenBlocked, after.tokenBlocked);
	tokenBranchNumber = ValueTracker<int>(before.tokenBranchNumber, after.tokenBranchNumber);
	metadata = ValueTracker<CellMetadata>(before.metadata, after.metadata);
	cellFeatures = ValueTracker<CellFeatureDescription>(before.cellFeature, after.cellFeature);
	tokens = ValueTracker<vector<TokenDescription>>(before.tokens, after.tokens);
    tokenUsages = ValueTracker<int>(before.tokenUsages, after.tokenUsages);
}

bool CellChangeDescription::isEmpty() const
{
    return !pos && !energy && !maxConnections && !connectingCells && !tokenBlocked && !tokenBranchNumber && !metadata
        && !cellFeatures && !tokens;
}

ParticleChangeDescription::ParticleChangeDescription(ParticleDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	vel = desc.vel;
	energy = desc.energy;
	metadata = desc.metadata;
}

ParticleChangeDescription::ParticleChangeDescription(ParticleDescription const & before, ParticleDescription const & after)
{
	id = after.id;
	pos = ValueTracker<QVector2D>(before.pos, after.pos);
	vel = ValueTracker<QVector2D>(before.vel, after.vel);
	energy = ValueTracker<double>(before.energy, after.energy);
	metadata = ValueTracker<ParticleMetadata>(before.metadata, after.metadata);
}

bool ParticleChangeDescription::isEmpty() const
{
    return !pos && !vel && !energy && !metadata;
}

DataChangeDescription::DataChangeDescription(DataDescription const & desc)
{
	if (desc.clusters) {
		for (auto const& cluster : *desc.clusters) {
            for (auto const& [index, cell] : *cluster.cells | boost::adaptors::indexed(0)) {
                CellChangeDescription cellChange(cell);
                cellChange.vel = *cluster.vel;	//TODO remove when DataDescription has new model
                addNewCell(cellChange);
            }
		}
        completeConnections();
	}
	if (desc.particles) {
		for (auto const& particle : *desc.particles) {
			addNewParticle(particle);
		}
	}
}

DataChangeDescription::DataChangeDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
    //TODO remove when DataDescription has new model
    std::unordered_map<uint64_t, QVector2D> cellVelByIdBefore;
    std::unordered_map<uint64_t, QVector2D> cellVelByIdAfter;
    if (dataBefore.clusters) {
        for (auto const& cluster : *dataBefore.clusters) {
            for (auto const& cell : *cluster.cells) {
                cellVelByIdBefore.insert_or_assign(cell.id, *cluster.vel);
            }
        }
    }
    if (dataAfter.clusters) {
        for (auto const& cluster : *dataAfter.clusters) {
            for (auto const& cell : *cluster.cells) {
                cellVelByIdAfter.insert_or_assign(cell.id, *cluster.vel);
            }
        }
    }
    //---

	if (dataBefore.clusters && dataAfter.clusters) {
        std::vector<CellDescription> cellsBefore;
        std::vector<CellDescription> cellsAfter;
        for (auto const& cluster : *dataBefore.clusters) {
            cellsBefore.insert(cellsBefore.begin(), cluster.cells->begin(), cluster.cells->end());
        }
        for (auto const& cluster : *dataAfter.clusters) {
            cellsAfter.insert(cellsAfter.begin(), cluster.cells->begin(), cluster.cells->end());
        }

        unordered_map<uint64_t, int> cellsAfterIndicesByIds;
        for (int index = 0; index < cellsAfter.size(); ++index) {
            cellsAfterIndicesByIds.insert_or_assign(cellsAfter.at(index).id, index);
		}

		for (auto const& cellBefore : cellsBefore) {
            auto cellIdAfterIt = cellsAfterIndicesByIds.find(cellBefore.id);
            if (cellIdAfterIt == cellsAfterIndicesByIds.end()) {
                addDeletedCell(CellChangeDescription().setId(cellBefore.id).setPos(*cellBefore.pos));
			}
			else {
				int cellAfterIndex = cellIdAfterIt->second;
                auto const& cellAfter = cellsAfter.at(cellAfterIndex);
				CellChangeDescription change(cellBefore, cellAfter);
				if (!change.isEmpty()) {
					//TODO remove when DataDescription has new model
                    change.vel = ValueTracker<QVector2D>(
                        cellVelByIdBefore.at(cellBefore.id), cellVelByIdAfter.at(cellAfter.id));
					//---
					addModifiedCell(change);
				}
				cellsAfterIndicesByIds.erase(cellBefore.id);
			}
		}

		for (auto const& cellAfterIndex : cellsAfterIndicesByIds | boost::adaptors::map_values) {
            auto const& cellAfter = cellsAfter.at(cellAfterIndex);
            CellChangeDescription change(cellAfter);
            //TODO remove when DataDescription has new model
            change.vel = cellVelByIdAfter.at(cellAfter.id);
            //---
            addNewCell(change);
		}
	}
	if (!dataBefore.clusters && dataAfter.clusters) {
		for (auto const& clusterAfter : *dataAfter.clusters) {
            for (auto const& cellAfter : *clusterAfter.cells) {
                CellChangeDescription change(cellAfter);
                //TODO remove when DataDescription has new model
                change.vel = cellVelByIdAfter.at(cellAfter.id);
                //---
                addNewCell(change);
            }
		}
	}
    completeConnections();

	if (dataBefore.particles && dataAfter.particles) {
		unordered_map<uint64_t, int> particleAfterIndicesByIds;
		for (int index = 0; index < dataAfter.particles->size(); ++index) {
			particleAfterIndicesByIds.insert_or_assign(dataAfter.particles->at(index).id, index);
		}

		for (auto const& particleBefore : *dataBefore.particles) {
			auto particleIdAfterIt = particleAfterIndicesByIds.find(particleBefore.id);
			if (particleIdAfterIt == particleAfterIndicesByIds.end()) {
				addDeletedParticle(ParticleChangeDescription().setId(particleBefore.id).setPos(*particleBefore.pos));
			}
			else {
				int particleAfterIndex = particleIdAfterIt->second;
				auto const& particleAfter = dataAfter.particles->at(particleAfterIndex);
				ParticleChangeDescription change(particleBefore, particleAfter);
				if (!change.isEmpty()) {
					addModifiedParticle(change);
				}
				particleAfterIndicesByIds.erase(particleBefore.id);
			}
		}

		for (auto const& particleAfterIndexById : particleAfterIndicesByIds) {
			auto const& particleAfter = dataAfter.particles->at(particleAfterIndexById.second);
			addNewParticle(ParticleChangeDescription(particleAfter));
		}
	}
	if (!dataBefore.particles && dataAfter.particles) {
		for (auto const& particleAfter : *dataAfter.particles) {
			addNewParticle(ParticleChangeDescription(particleAfter));
		}
	}
}

void DataChangeDescription::completeConnections()
{
    std::unordered_map<uint64_t, int> cellIndexById;
    for (auto const& [index, cell] : cells | boost::adaptors::indexed(0)) {
        if (cell.isDeleted()) {
			continue;
        }
        cellIndexById.insert_or_assign(cell->id, index);
    }
    for (auto& cell : cells) {
        std::list<ConnectionChangeDescription> connections;
        for (auto& connectingCell : *cell->connectingCells) {
            auto const& connectingCellDesc = cells.at(cellIndexById.at(connectingCell.cellId));
            connectingCell.distance = (*connectingCellDesc->pos - *cell->pos).length();
            //TODO angleToPrevious
        }
    }
}

