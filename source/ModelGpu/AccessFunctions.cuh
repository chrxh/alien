#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"

#include "SimulationData.cuh"

__device__ void getClusterAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
	SimulationData const& data, SimulationAccessTO& simulationTO, int clusterIndex)
{
	Cluster const& cluster = data.clustersAC1.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	calcPartition(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	__shared__ bool containedInRect;
	__shared__ CellAccessTO* cellTOs;
	if (0 == threadIdx.x) {
		containedInRect = false;
	}
	__syncthreads();

	for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		Cell const& cell = cluster.cells[cellIndex];
		if (isContained(rectUpperLeft, rectLowerRight, cell.absPos)) {
			containedInRect = true;
		}
	}
	__syncthreads();

	if (containedInRect) {
		__shared__ int cellTOIndex;
		if (0 == threadIdx.x) {
			int clusterAccessIndex = atomicAdd(simulationTO.numClusters, 1);
			cellTOIndex = atomicAdd(simulationTO.numCells, cluster.numCells);

			cellTOs = &simulationTO.cells[cellTOIndex];

			ClusterAccessTO& clusterTO = simulationTO.clusters[clusterAccessIndex];
			clusterTO.id = cluster.id;
			clusterTO.pos = cluster.pos;
			clusterTO.vel = cluster.vel;
			clusterTO.angle = cluster.angle;
			clusterTO.angularVel = cluster.angularVel;
			clusterTO.numCells = cluster.numCells;
			clusterTO.cellStartIndex = cellTOIndex;
		}
		__syncthreads();

		for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			Cell const& cell = cluster.cells[cellIndex];
			cellTOs[cellIndex].id = cell.id;
			cellTOs[cellIndex].pos = cell.absPos;
			cellTOs[cellIndex].energy = cell.energy;
			cellTOs[cellIndex].maxConnections = cell.maxConnections;
			cellTOs[cellIndex].numConnections = cell.numConnections;
			for (int i = 0; i < cell.numConnections; ++i) {
				int connectingCellIndex = cell.connections[i] - cluster.cells + cellTOIndex;
				cellTOs[cellIndex].connectionIndices[i] = connectingCellIndex;
			}
		}
	}
}

__device__ void getParticleAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
	SimulationData const& data, SimulationAccessTO& access, int particleIndex)
{
	Particle const& particle = data.particlesAC1.getEntireArray()[particleIndex];
	if (particle.pos.x >= rectUpperLeft.x
		&& particle.pos.x <= rectLowerRight.x
		&& particle.pos.y >= rectUpperLeft.y
		&& particle.pos.y <= rectLowerRight.y)
	{
		int particleAccessIndex = atomicAdd(access.numParticles, 1);
		ParticleAccessTO& particleAccess = access.particles[particleAccessIndex];

		particleAccess.id = particle.id;
		particleAccess.pos = particle.pos;
		particleAccess.vel = particle.vel;
		particleAccess.energy = particle.energy;
	}
}

__global__ void getSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
	SimulationData data, SimulationAccessTO access)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		getClusterAccessData(rectUpperLeft, rectLowerRight, data, access, clusterIndex);
	}

	indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	numEntities = data.particlesAC1.getNumEntries();

	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		getParticleAccessData(rectUpperLeft, rectLowerRight, data, access, particleIndex);
	}

}

__device__ void filterClusterData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
	SimulationData& data, int clusterIndex)
{
	Cluster const& cluster = data.clustersAC1.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	calcPartition(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	__shared__ bool containedInRect;
	if (0 == threadIdx.x) {
		containedInRect = false;
	}
	__syncthreads();

	for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		Cell const& cell = cluster.cells[cellIndex];
		if (isContained(rectUpperLeft, rectLowerRight, cell.absPos)) {
			containedInRect = true;
		}
	}
	__syncthreads();

	if (!containedInRect) {
		__shared__ Cell* newCells;
		__shared__ Cluster* newCluster;
		if (0 == threadIdx.x) {
			newCluster = data.clustersAC2.getNewElement();
			newCells = data.cellsAC2.getNewSubarray(cluster.numCells);
			*newCluster = cluster;
			newCluster->cells = newCells;
		}
		__syncthreads();

		for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			Cell& cell = cluster.cells[cellIndex];
			newCells[cellIndex] = cell;
			cell.cluster = newCluster;
			cell.nextTimestep = &newCells[cellIndex];
		}
		__syncthreads();

		for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			Cell& cell = cluster.cells[cellIndex];
			int numConnections = cell.numConnections;
			for (int i = 0; i < numConnections; ++i) {
				cell.connections[i] = cell.connections[i]->nextTimestep;
			}
		}
	}
}

__device__ void filterParticleData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
	SimulationData& data, int particleIndex)
{
	Particle const& particle = data.particlesAC1.getEntireArray()[particleIndex];
	if (!isContained(rectUpperLeft, rectLowerRight, particle.pos)) {
		Particle* newParticle = data.particlesAC2.getNewElement();
		*newParticle = particle;
	}
}

class EntityConverter
{
private:
	BasicMap _map;
	SimulationData* _data;
	SimulationAccessTO const* _simulationTO;

public:
	__device__ void init(SimulationData* data, SimulationAccessTO const* simulationTO)
	{
		_data = data;
		_simulationTO = simulationTO;
		_map.init(data->size);
	}

	__device__ void createClusterFromTO(ClusterAccessTO const& clusterTO)
	{
		__shared__ Cluster* cluster;
		__shared__ float angularMass;
		__shared__ float invRotMatrix[2][2];
		if (0 == threadIdx.x) {
			cluster = _data->clustersAC2.getNewElement();
			cluster->id = clusterTO.id;
			cluster->pos = clusterTO.pos;
			cluster->vel = clusterTO.vel;
			cluster->angle = clusterTO.angle;
			cluster->angularVel = clusterTO.angularVel;
			cluster->numCells = clusterTO.numCells;
			cluster->cells = _data->cellsAC2.getNewSubarray(cluster->numCells);

			cluster->decompositionRequired = false;
			cluster->locked = 0;
			cluster->clusterToFuse = nullptr;

			angularMass = 0.0f;
			Physics::inverseRotationMatrix(cluster->angle, invRotMatrix);
		}
		__syncthreads();

		int startCellIndex;
		int endCellIndex;
		calcPartition(cluster->numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

		for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			Cell& cell = cluster->cells[cellIndex];
			CellAccessTO const& cellTO =  _simulationTO->cells[clusterTO.cellStartIndex + cellIndex];
			atomicAdd(&angularMass, lengthSquared(cell.relPos));
			cell.id = cellTO.id;
			cell.cluster = cluster;
			cell.absPos = cellTO.pos;

			float2 deltaPos = sub(cell.absPos, cluster->pos);
			cell.relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
			cell.relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];
		
			auto r = sub(cell.absPos, cluster->pos);
			_map.mapDisplacementCorrection(r);
			cell.vel = Physics::tangentialVelocity(r, cluster->vel, cluster->angularVel);
			
			cell.energy = cellTO.energy;
			cell.maxConnections = cellTO.maxConnections;
			cell.numConnections = cellTO.numConnections;
			for (int i = 0; i < cell.numConnections; ++i) {
				int index = cellTO.connectionIndices[i] - clusterTO.cellStartIndex;
				cell.connections[i] = cluster->cells + index;
			}

			cell.nextTimestep = nullptr;
			cell.protectionCounter = 0;
			cell.alive = true;
		}
		__syncthreads();

		if (0 == threadIdx.x) {
			cluster->angularMass = angularMass;
		}
	}

	__device__ void createParticleFromTO(ParticleAccessTO const& particleTO)
	{
		Particle* particle = _data->particlesAC2.getNewElement();
		particle->id = particleTO.id;
		particle->pos = particleTO.pos;
		particle->vel = particleTO.vel;
		particle->energy = particleTO.energy;
		particle->locked = 0;
		particle->alive = true;
	}
};

__global__ void filterData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		filterClusterData(rectUpperLeft, rectLowerRight, data, clusterIndex);
	}

	indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	numEntities = data.particlesAC1.getNumEntries();
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		filterParticleData(rectUpperLeft, rectLowerRight, data, particleIndex);
	}
	__syncthreads();
}


__device__ void convertData(SimulationData data, SimulationAccessTO const& simulationTO)
{
	__shared__ EntityConverter converter;
	if (0 == threadIdx.x) {
		converter.init(&data, &simulationTO);
	}
	__syncthreads();

	int indexResource = blockIdx.x;
	int numEntities = *simulationTO.numClusters;
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		converter.createClusterFromTO(simulationTO.clusters[clusterIndex]);
	}

	numEntities = *simulationTO.numParticles;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		converter.createParticleFromTO(simulationTO.particles[particleIndex]);
	}
}

__global__ void setSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
	SimulationData data, SimulationAccessTO access)
{
	convertData(data, access);
}
