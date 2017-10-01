#include "ChangeDescriptions.h"

#define SET_DELTA(before, after, delta)\
	if(before && after) {\
		if (*before != *after) { \
			delta = *after; \
		} \
	} \
	if(!before && after) { \
		delta = *after; \
	}

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
	cellFunction = desc.cellFeature;
	tokens = desc.tokens;
}

CellChangeDescription::CellChangeDescription(CellDescription const & before, CellDescription const & after)
{
	id = after.id;
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.energy, after.energy, energy);
	SET_DELTA(before.maxConnections, after.maxConnections, maxConnections);
	SET_DELTA(before.connectingCells, after.connectingCells, connectingCells);
	SET_DELTA(before.tokenBlocked, after.tokenBlocked, tokenBlocked);
	SET_DELTA(before.tokenBranchNumber, after.tokenBranchNumber, tokenBranchNumber);
	SET_DELTA(before.metadata, after.metadata, metadata);
	SET_DELTA(before.cellFeature, after.cellFeature, cellFunction);
	SET_DELTA(before.tokens, after.tokens, tokens);
}

bool CellChangeDescription::isEmpty() const
{
	return !pos.is_initialized()
		&& !energy.is_initialized()
		&& !maxConnections.is_initialized()
		&& !connectingCells.is_initialized()
		&& !tokenBlocked.is_initialized()
		&& !tokenBranchNumber.is_initialized()
		&& !metadata.is_initialized()
		&& !cellFunction.is_initialized()
		&& !tokens.is_initialized()
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
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.vel, after.vel, vel);
	SET_DELTA(before.angle, after.angle, angle);
	SET_DELTA(before.angularVel, after.angularVel, angularVel);
	SET_DELTA(before.metadata, after.metadata, metadata);

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
	_posBefore = before.pos;
}

bool ClusterChangeDescription::isEmpty() const
{
	return !pos.is_initialized()
		&& !vel.is_initialized()
		&& !angle.is_initialized()
		&& !angularVel.is_initialized()
		&& !metadata.is_initialized()
		&& cells.empty()
		;
}

QVector2D ClusterChangeDescription::getPosBefore() const
{
	return *_posBefore;
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
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.vel, after.vel, vel);
	SET_DELTA(before.energy, after.energy, energy);
	SET_DELTA(before.metadata, after.metadata, metadata);
	_posBefore = before.pos;
}

bool ParticleChangeDescription::isEmpty() const
{
	return !pos.is_initialized()
		&& !vel.is_initialized()
		&& !energy.is_initialized()
		&& !metadata.is_initialized()
		;
}

QVector2D ParticleChangeDescription::getPosBefore() const
{
	return *_posBefore;
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


