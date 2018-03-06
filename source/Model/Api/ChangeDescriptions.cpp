#include "ChangeDescriptions.h"

CellChangeDescription::CellChangeDescription(CellDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	energy = desc.energy;
	maxConnections = desc.maxConnections;
	connectingCells = desc.connectingCells;
	tokenBlocked = desc.tokenBlocked;
	tokenBranchNumber = desc.tokenBranchNumber;
	metadata = desc.metadata;
	cellFeatures = desc.cellFeature;
	tokens = desc.tokens;
}

CellChangeDescription::CellChangeDescription(CellDescription const & before, CellDescription const & after)
{
	id = after.id;
	pos = ValueTracker<QVector2D>(before.pos, after.pos);
	energy = ValueTracker<double>(before.energy, after.energy);
	maxConnections = ValueTracker<int>(before.maxConnections, after.maxConnections);
	connectingCells = ValueTracker<list<uint64_t>>(before.connectingCells, after.connectingCells);
	tokenBlocked = ValueTracker<bool>(before.tokenBlocked, after.tokenBlocked);
	tokenBranchNumber = ValueTracker<int>(before.tokenBranchNumber, after.tokenBranchNumber);
	metadata = ValueTracker<CellMetadata>(before.metadata, after.metadata);
	cellFeatures = ValueTracker<CellFeatureDescription>(before.cellFeature, after.cellFeature);
	tokens = ValueTracker<vector<TokenDescription>>(before.tokens, after.tokens);
}

bool CellChangeDescription::isEmpty() const
{
	return !pos
		&& !energy
		&& !maxConnections
		&& !connectingCells
		&& !tokenBlocked
		&& !tokenBranchNumber
		&& !metadata
		&& !cellFeatures
		&& !tokens
		;
}

ClusterChangeDescription::ClusterChangeDescription(ClusterDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	vel = desc.vel;
	angle = desc.angle;
	angularVel = desc.angularVel;
	metadata = desc.metadata;
	if (desc.cells) {
		for (auto const& cell : *desc.cells) {
			addNewCell(cell);
		}
	}
}

ClusterChangeDescription::ClusterChangeDescription(ClusterDescription const & before, ClusterDescription const & after)
{
	id = after.id;
	pos = ValueTracker<QVector2D>(before.pos, after.pos);
	vel = ValueTracker<QVector2D>(before.vel, after.vel);
	angle = ValueTracker<double>(before.angle, after.angle);
	angularVel = ValueTracker<double>(before.angularVel, after.angularVel);
	metadata = ValueTracker<ClusterMetadata>(before.metadata, after.metadata);

	if (before.cells && after.cells) {
		unordered_map<uint64_t, int> cellAfterIndicesByIds;
		for (int index = 0; index < after.cells->size(); ++index) {
			cellAfterIndicesByIds.insert_or_assign(after.cells->at(index).id, index);
		}

		for (auto const& cellBefore : *before.cells) {
			auto cellIdAfterIt = cellAfterIndicesByIds.find(cellBefore.id);
			if (cellIdAfterIt == cellAfterIndicesByIds.end()) {
				addDeletedCell(CellChangeDescription().setId(cellBefore.id).setPos(*cellBefore.pos));
			}
			else {
				int cellAfterIndex = cellIdAfterIt->second;
				auto const& cellAfter = after.cells->at(cellAfterIndex);
				CellChangeDescription change(cellBefore, cellAfter);
				if (!change.isEmpty()) {
					addModifiedCell(change);
				}
				cellAfterIndicesByIds.erase(cellAfter.id);
			}
		}

		for (auto const& cellAfterIndexById : cellAfterIndicesByIds) {
			auto const& cellAfter = after.cells->at(cellAfterIndexById.second);
			addNewCell(CellChangeDescription(cellAfter));
		}
	}
	if (!before.cells && after.cells) {
		for (auto const& cellAfter : *after.cells) {
			addNewCell(CellChangeDescription(cellAfter));
		}
	}
}

bool ClusterChangeDescription::isEmpty() const
{
	return !pos
		&& !vel
		&& !angle
		&& !angularVel
		&& !metadata
		&& cells.empty()
		;
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
	return !pos
		&& !vel
		&& !energy
		&& !metadata
		;
}

DataChangeDescription::DataChangeDescription(DataDescription const & desc)
{
	if (desc.clusters) {
		for (auto const& cluster : *desc.clusters) {
			addNewCluster(cluster);
		}
	}
	if (desc.particles) {
		for (auto const& particle : *desc.particles) {
			addNewParticle(particle);
		}
	}
}

DataChangeDescription::DataChangeDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
	if (dataBefore.clusters && dataAfter.clusters) {
		unordered_map<uint64_t, int> clusterAfterIndicesByIds;
		for (int index = 0; index < dataAfter.clusters->size(); ++index) {
			clusterAfterIndicesByIds.insert_or_assign(dataAfter.clusters->at(index).id, index);
		}

		for (auto const& clusterBefore : *dataBefore.clusters) {
			auto clusterIdAfterIt = clusterAfterIndicesByIds.find(clusterBefore.id);
			if (clusterIdAfterIt == clusterAfterIndicesByIds.end()) {
				addDeletedCluster(ClusterChangeDescription().setId(clusterBefore.id).setPos(*clusterBefore.pos));
			}
			else {
				int clusterAfterIndex = clusterIdAfterIt->second;
				auto const& clusterAfter = dataAfter.clusters->at(clusterAfterIndex);
				ClusterChangeDescription change(clusterBefore, clusterAfter);
				if (!change.isEmpty()) {
					addModifiedCluster(change);
				}
				clusterAfterIndicesByIds.erase(clusterBefore.id);
			}
		}

		for (auto const& clusterAfterIndexById : clusterAfterIndicesByIds) {
			auto const& clusterAfter = dataAfter.clusters->at(clusterAfterIndexById.second);
			addNewCluster(ClusterChangeDescription(clusterAfter));
		}
	}
	if (!dataBefore.clusters && dataAfter.clusters) {
		for (auto const& clusterAfter : *dataAfter.clusters) {
			addNewCluster(ClusterChangeDescription(clusterAfter));
		}
	}

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


