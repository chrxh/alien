#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ClusterProcessorOnCopyData
{
public:
	__inline__ __device__ void init(SimulationData& data, int clusterIndex);

	__inline__ __device__ void processingDecomposition();
	__inline__ __device__ void processingClusterCopy();

private:
	__inline__ __device__ void copyClusterWithDecomposition();
	__inline__ __device__ void copyClusterWithoutDecompositionAndFusion();
	__inline__ __device__ void copyClusterWithFusion();
	__inline__ __device__ void copyTokens(Cluster* sourceCluster, Cluster* targetCluster);

	__inline__ __device__ void setSuccessorCell(Cell* origCell, Cell* newCell, Cluster* newCluster);
	__inline__ __device__ void correctCellConnections(Cell* origCell);
	__inline__ __device__ void correctTokens(Token* token);

	SimulationData* _data;
	Map<Cell> _cellMap;

	Cluster *_origCluster;

	int _startCellIndex;
	int _endCellIndex;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ClusterProcessorOnCopyData::init(SimulationData& data, int clusterIndex)
{
	_data = &data;
	_origCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
	_cellMap.init(data.size, data.cellMap);

	calcPartition(_origCluster->numCells, threadIdx.x, blockDim.x, _startCellIndex, _endCellIndex);

	__syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyClusterWithDecomposition()
{
	__shared__ int numDecompositions;
	struct Entry {
		int tag;
		float invRotMatrix[2][2];
		Cluster cluster;
	};
	__shared__ Entry entries[MAX_DECOMPOSITIONS];
	if (0 == threadIdx.x) {
		numDecompositions = 0;
		for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
			entries[i].tag = -1;
			entries[i].cluster.pos = { 0.0f, 0.0f };
			entries[i].cluster.vel = { 0.0f, 0.0f };
			entries[i].cluster.angle = _origCluster->angle;
			entries[i].cluster.angularVel = 0.0f;
			entries[i].cluster.angularMass = 0.0f;
			entries[i].cluster.numCells = 0;
			entries[i].cluster.numTokens = 0;
			entries[i].cluster.decompositionRequired = false;
			entries[i].cluster.clusterToFuse = nullptr;
			entries[i].cluster.locked = 0;
		}
	}
	__syncthreads();
	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell& cell = _origCluster->cells[cellIndex];
		if (!cell.alive) {
			continue;
		}
		bool foundMatch = false;
		for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
			int origTag = atomicCAS(&entries[index].tag, -1, cell.tag);
			if (-1 == origTag) {	//use free 
				atomicAdd(&numDecompositions, 1);
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);
				atomicAdd(&entries[index].cluster.vel.x, cell.vel.x);
				atomicAdd(&entries[index].cluster.vel.y, cell.vel.y);

				entries[index].cluster.id = _data->numberGen.createNewId_kernel();
				Physics::inverseRotationMatrix(entries[index].cluster.angle, entries[index].invRotMatrix);
				foundMatch = true;
				break;
			}
			if (cell.tag == origTag) {	//matching entry
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);
				atomicAdd(&entries[index].cluster.vel.x, cell.vel.x);
				atomicAdd(&entries[index].cluster.vel.y, cell.vel.y);

				foundMatch = true;
				break;
			}
		}
		if (!foundMatch) {	//no match? use last entry
			cell.tag = entries[MAX_DECOMPOSITIONS - 1].tag;
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.numCells, 1);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.x, cell.absPos.x);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.y, cell.absPos.y);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.x, cell.vel.x);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.y, cell.vel.y);

			entries[MAX_DECOMPOSITIONS - 1].cluster.decompositionRequired = true;
		}
	}
	__syncthreads();

	int startDecompositionIndex;
	int endDecompositionIndex;
	__shared__ Cluster* newClusters[MAX_DECOMPOSITIONS];
	calcPartition(numDecompositions, threadIdx.x, blockDim.x, startDecompositionIndex, endDecompositionIndex);
	for (int index = startDecompositionIndex; index <= endDecompositionIndex; ++index) {
		auto numCells = entries[index].cluster.numCells;
		entries[index].cluster.pos.x /= numCells;
		entries[index].cluster.pos.y /= numCells;
		entries[index].cluster.vel.x /= numCells;
		entries[index].cluster.vel.y /= numCells;
		entries[index].cluster.cells = _data->cellsAC2.getNewSubarray(numCells);
		entries[index].cluster.numCells = 0;

		newClusters[index] = _data->clustersAC2.getNewElement();
		*newClusters[index] = entries[index].cluster;
	}
	__syncthreads();

	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell& cell = _origCluster->cells[cellIndex];
		for (int index = 0; index < numDecompositions; ++index) {
			if (cell.tag == entries[index].tag) {
				Cluster* newCluster = newClusters[index];
				float2 deltaPos = sub(cell.absPos, newCluster->pos);
				_cellMap.mapDisplacementCorrection(deltaPos);
				auto angularMass = deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y;
				atomicAdd(&newCluster->angularMass, angularMass);

				auto r = sub(cell.absPos, _origCluster->pos);
				_cellMap.mapDisplacementCorrection(r);
				float2 relVel = sub(cell.vel, newCluster->vel);
				float angularMomentum = Physics::angularMomentum(r, relVel);
				atomicAdd(&newCluster->angularVel, angularMomentum);

				float(&invRotMatrix)[2][2] = entries[index].invRotMatrix;
				cell.relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
				cell.relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];

				int newCellIndex = atomicAdd(&newCluster->numCells, 1);
				Cell& newCell = newCluster->cells[newCellIndex];
				setSuccessorCell(&cell, &newCell, newCluster);
			}
		}
	}
	__syncthreads();

	for (int index = startDecompositionIndex; index <= endDecompositionIndex; ++index) {
		Cluster* newCluster = newClusters[index];

		//newCluster->angularVel contains angular momentum until here
		newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);
	}
	for (int index = 0; index < numDecompositions; ++index) {
		Cluster* newCluster = newClusters[index];
		copyTokens(_origCluster, newCluster);
	}

	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell& cell = _origCluster->cells[cellIndex];
		if (!cell.alive) {
			continue;
		}
		correctCellConnections(cell.nextTimestep);
	}
	__syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyClusterWithoutDecompositionAndFusion()
{
	__shared__ Cluster* newCluster;
	__shared__ Cell* newCells;

	if (threadIdx.x == 0) {
		newCluster = _data->clustersAC2.getNewElement();
		*newCluster = *_origCluster;
		newCells = _data->cellsAC2.getNewSubarray(_origCluster->numCells);
		newCluster->cells = newCells;
	}
	__syncthreads();

	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell *origCell = &_origCluster->cells[cellIndex];
		Cell *newCell = &newCells[cellIndex];
		setSuccessorCell(origCell, newCell, newCluster);
	}

	__syncthreads();
	
	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		correctCellConnections(&newCells[cellIndex]);
	}

	int startTokenIndex;
	int endTokenIndex;
	calcPartition(_origCluster->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);
	for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
		correctTokens(&_origCluster->tokens[tokenIndex]);
	}
	__syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyClusterWithFusion()
{
	if (_origCluster < _origCluster->clusterToFuse) {
		__shared__ Cluster* newCluster;
		__shared__ Cluster* otherCluster;
		__shared__ float2 correction;
		if (0 == threadIdx.x) {
			otherCluster = _origCluster->clusterToFuse;
			newCluster = _data->clustersAC2.getNewElement();
			newCluster->id = _origCluster->id;
			newCluster->angle = 0.0f;
			newCluster->numCells = _origCluster->numCells + otherCluster->numCells;
			newCluster->numTokens = 0;
			newCluster->decompositionRequired = _origCluster->decompositionRequired || otherCluster->decompositionRequired;
			newCluster->locked = 0;
			newCluster->clusterToFuse = nullptr;
			newCluster->cells = _data->cellsAC2.getNewSubarray(newCluster->numCells);

			correction = _cellMap.correctionIncrement(_origCluster->pos, otherCluster->pos);	//to be added to otherCluster

			newCluster->pos =
				div(add(mul(_origCluster->pos, _origCluster->numCells), mul(add(otherCluster->pos, correction), otherCluster->numCells)),
					newCluster->numCells);
			newCluster->vel = div(
				add(mul(_origCluster->vel, _origCluster->numCells), mul(otherCluster->vel, otherCluster->numCells)), newCluster->numCells);
			newCluster->angularVel = 0.0f;
			newCluster->angularMass = 0.0f;
		}
		__syncthreads();

		for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
			Cell* newCell = &newCluster->cells[cellIndex];
			Cell* origCell = &_origCluster->cells[cellIndex];
			setSuccessorCell(origCell, newCell, newCluster);
			auto relPos = sub(newCell->absPos, newCluster->pos);
			newCell->relPos = relPos;
			newCell->cluster = newCluster;
			atomicAdd(&newCluster->angularMass, lengthSquared(relPos));

			auto r = sub(newCell->absPos, _origCluster->pos);
			_cellMap.mapDisplacementCorrection(r);
			float2 relVel = sub(newCell->vel, _origCluster->vel);
			float angularMomentum = Physics::angularMomentum(r, relVel);
			atomicAdd(&newCluster->angularVel, angularMomentum);
		}
		__syncthreads();

		int startOtherCellIndex;
		int endOtherCellIndex;
		calcPartition(otherCluster->numCells, threadIdx.x, blockDim.x, startOtherCellIndex, endOtherCellIndex);

		for (int otherCellIndex = startOtherCellIndex; otherCellIndex <= endOtherCellIndex; ++otherCellIndex) {
			Cell* newCell = &newCluster->cells[_origCluster->numCells + otherCellIndex];
			Cell* origCell = &otherCluster->cells[otherCellIndex];
			setSuccessorCell(origCell, newCell, newCluster);

			auto r = sub(newCell->absPos, otherCluster->pos);
			_cellMap.mapDisplacementCorrection(r);

			newCell->absPos = add(newCell->absPos, correction);
			auto relPos = sub(newCell->absPos, newCluster->pos);
			newCell->relPos = relPos;
			newCell->cluster = newCluster;
			atomicAdd(&newCluster->angularMass, lengthSquared(relPos));

			float2 relVel = sub(newCell->vel, otherCluster->vel);
			float angularMomentum = Physics::angularMomentum(r, relVel);
			atomicAdd(&newCluster->angularVel, angularMomentum);
		}
		__syncthreads();

		for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
			correctCellConnections(&newCluster->cells[cellIndex]);
		}
		for (int otherCellIndex = startOtherCellIndex; otherCellIndex <= endOtherCellIndex; ++otherCellIndex) {
			correctCellConnections(&newCluster->cells[_origCluster->numCells + otherCellIndex]);
		}

		if (0 == threadIdx.x) {
			newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);
		}

		copyTokens(_origCluster, newCluster);
		copyTokens(_origCluster->clusterToFuse, newCluster);
	}
	else {
		//do not copy anything
	}
	__syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyTokens(Cluster* sourceCluster, Cluster* targetCluster)
{
	__shared__ int numberOfTokensToCopy;
	if (0 == threadIdx.x) {
		numberOfTokensToCopy = 0;
	}
	__syncthreads();

	int startTokenIndex;
	int endTokenIndex;
	calcPartition(sourceCluster->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);

	for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
		auto const& token = sourceCluster->tokens[tokenIndex];
		auto const& cell = token.cell;
		auto const& successorCell = *cell->nextTimestep;
		auto const& successorCluster = successorCell.cluster;
		if (successorCluster == targetCluster) {
			++numberOfTokensToCopy;
		}
	}
	__syncthreads();

	__shared__ int tokenCopyIndex;
	if (0 == threadIdx.x) {
		tokenCopyIndex = targetCluster->numTokens;
		auto newTokenArray = _data->tokensAC2.getNewSubarray(numberOfTokensToCopy);
		if (0 == targetCluster->numTokens) {
			targetCluster->tokens = newTokenArray;
		}
		targetCluster->numTokens += numberOfTokensToCopy;
	}
	__syncthreads();

	for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
		auto& token = sourceCluster->tokens[tokenIndex];
		auto& cell = token.cell;
		auto& successorCell = *cell->nextTimestep;
		auto& successorCluster = successorCell.cluster;
		if (successorCluster == targetCluster) {
			int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
			auto& targetToken = targetCluster->tokens[prevTokenCopyIndex];
			targetToken = token;
			targetToken.cell = &successorCell;
		}
	}
	__syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::processingClusterCopy()
{
	if (_origCluster->numCells == 1 && !_origCluster->cells[0].alive && !_origCluster->clusterToFuse) {
		__syncthreads();
		return;
	}

	if (_origCluster->decompositionRequired && !_origCluster->clusterToFuse) {
		copyClusterWithDecomposition();
	}
	else if (_origCluster->clusterToFuse) {
		copyClusterWithFusion();
	}
	else {
		copyClusterWithoutDecompositionAndFusion();
	}
}

__inline__ __device__ void ClusterProcessorOnCopyData::processingDecomposition()
{
	if (_origCluster->decompositionRequired && !_origCluster->clusterToFuse) {
		__shared__ bool changes;

		for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
			_origCluster->cells[cellIndex].tag = cellIndex;
		}
		do {
			changes = false;
			__syncthreads();
			for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
				Cell& cell = _origCluster->cells[cellIndex];
				if (cell.alive) {
					for (int i = 0; i < cell.numConnections; ++i) {
						Cell& otherCell = *cell.connections[i];
						if (otherCell.alive) {
							if (otherCell.tag < cell.tag) {
								cell.tag = otherCell.tag;
								changes = true;
							}
						}
						else {
							for (int j = i + 1; j < cell.numConnections; ++j) {
								cell.connections[j - 1] = cell.connections[j];
							}
							--cell.numConnections;
							--i;
						}
					}
				}
				else {
					cell.numConnections = 0;
				}
			}
			__syncthreads();
		} while (changes);
	}
}

__inline__ __device__  void ClusterProcessorOnCopyData::setSuccessorCell(Cell *origCell, Cell* newCell, Cluster* newCluster)
{
	*newCell = *origCell;
	newCell->cluster = newCluster;
	origCell->nextTimestep = newCell;
}

__inline__ __device__ void ClusterProcessorOnCopyData::correctCellConnections(Cell* cell)
{
	int numConnections = cell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		cell->connections[i] = cell->connections[i]->nextTimestep;
	}
}

__inline__ __device__ void ClusterProcessorOnCopyData::correctTokens(Token* token)
{
	token->cell = token->cell->nextTimestep;
}


