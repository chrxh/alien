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
	cellFunction = desc.cellFunction;
	tokens = desc.tokens;
}

CellChangeDescription::CellChangeDescription(CellDescription const & before, CellDescription const & after)
{
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.energy, after.energy, energy);
	SET_DELTA(before.maxConnections, after.maxConnections, maxConnections);
	SET_DELTA(before.connectingCells, after.connectingCells, connectingCells);
	SET_DELTA(before.tokenBlocked, after.tokenBlocked, tokenBlocked);
	SET_DELTA(before.tokenBranchNumber, after.tokenBranchNumber, tokenBranchNumber);
	SET_DELTA(before.metadata, after.metadata, metadata);
	SET_DELTA(before.cellFunction, after.cellFunction, cellFunction);
	SET_DELTA(before.tokens, after.tokens, tokens);
}

ClusterChangeDescription::ClusterChangeDescription(ClusterDescription const & desc)
{
	id = desc.id;
	pos = desc.pos;
	vel = desc.vel;
	angle = desc.angle;
	angularVel = desc.angularVel;
	metadata = desc.metadata;
	for (auto const& cell : desc.cells) {
		addNewCell(cell);
	}
}

ClusterChangeDescription::ClusterChangeDescription(ClusterDescription const & before, ClusterDescription const & after)
{
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.vel, after.vel, vel);
	SET_DELTA(before.angle, after.angle, angle);
	SET_DELTA(before.angularVel, after.angularVel, angularVel);
	SET_DELTA(before.metadata, after.metadata, metadata);

	unordered_map<uint64_t, int> cellAfterIndicesByIds;
	for (int index = 0; index < after.cells.size(); ++index) {
		cellAfterIndicesByIds.insert_or_assign(after.cells.at(index).id, index);
	}

	for (auto const& cellBefore : after.cells) {
		auto cellIdAfterIt = cellAfterIndicesByIds.find(cellBefore.id);
		if (cellIdAfterIt == cellAfterIndicesByIds.end()) {
			addDeletedCell(cellBefore.id);
		}
		else {
			int cellAfterIndex = cellIdAfterIt->second;
			auto const& cellAfter = after.cells.at(cellAfterIndex);
			addModifiedCell(CellChangeDescription(cellBefore, cellAfter));
			cellAfterIndicesByIds.erase(cellBefore.id);
		}
	}

	for (auto const& cellAfterIndexById : cellAfterIndicesByIds) {
		auto const& cellAfter = after.cells.at(cellAfterIndexById.second);
		addNewCell(CellChangeDescription(cellAfter));
	}
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
	SET_DELTA(before.pos, after.pos, pos);
	SET_DELTA(before.vel, after.vel, vel);
	SET_DELTA(before.energy, after.energy, energy);
	SET_DELTA(before.metadata, after.metadata, metadata);
}

DataChangeDescription::DataChangeDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
	unordered_map<uint64_t, int> clusterAfterIndicesByIds;
	for (int index = 0; index < dataAfter.clusters.size(); ++index) {
		clusterAfterIndicesByIds.insert_or_assign(dataAfter.clusters.at(index).id, index);
	}

	for (auto const& clusterBefore : dataBefore.clusters) {
		auto clusterIdAfterIt = clusterAfterIndicesByIds.find(clusterBefore.id);
		if (clusterIdAfterIt == clusterAfterIndicesByIds.end()) {
			addDeletedCluster(clusterBefore.id);
		}
		else {
			int clusterAfterIndex = clusterIdAfterIt->second;
			auto const& clusterAfter = dataAfter.clusters.at(clusterAfterIndex);
			addModifiedCluster(ClusterChangeDescription(clusterBefore, clusterAfter));
			clusterAfterIndicesByIds.erase(clusterBefore.id);
		}
	}

	for (auto const& clusterAfterIndexById : clusterAfterIndicesByIds) {
		auto const& clusterAfter = dataAfter.clusters.at(clusterAfterIndexById.second);
		addNewCluster(ClusterChangeDescription(clusterAfter));
	}

	unordered_map<uint64_t, int> particleAfterIndicesByIds;
	for (int index = 0; index < dataAfter.particles.size(); ++index) {
		particleAfterIndicesByIds.insert_or_assign(dataAfter.particles.at(index).id, index);
	}

	for (auto const& particleBefore : dataBefore.particles) {
		auto particleIdAfterIt = particleAfterIndicesByIds.find(particleBefore.id);
		if (particleIdAfterIt == particleAfterIndicesByIds.end()) {
			addDeletedParticle(particleBefore.id);
		}
		else {
			int particleAfterIndex = particleIdAfterIt->second;
			auto const& particleAfter = dataAfter.particles.at(particleAfterIndex);
			addModifiedParticle(ParticleChangeDescription(particleBefore, particleAfter));
			particleAfterIndicesByIds.erase(particleBefore.id);
		}
	}

	for (auto const& particleAfterIndexById : particleAfterIndicesByIds) {
		auto const& particleAfter = dataAfter.particles.at(particleAfterIndexById.second);
		addNewParticle(ParticleChangeDescription(particleAfter));
	}
}

