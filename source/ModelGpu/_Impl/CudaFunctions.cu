#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "CudaFunctions.cuh"


#define DEG_TO_RAD 3.1415926535897932384626433832795/180.0
#define NUM_THREADS_PER_BLOCK 32
#define NUM_BLOCKS (32 * 5) /*160*/
#define NUM_CLUSTERS (NUM_BLOCKS * 50)
#define LAYERS 2

template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template<class T>
class ArrayController
{
private:
	int _size;
	int _lastEntry = 0;
	T* _data;

public:

	ArrayController() = default;
	ArrayController(int size)
		: _size(size)
	{
		cudaMallocManaged(&_data, sizeof(T) * size);
	}

	void free()
	{
		cudaFree(_data);
	}

	T* getArray(int size)
	{
		auto result = _lastEntry;
		_lastEntry += size;
		return &_data[result];
	}

	__device__ T* getArrayKernel(int size)
	{
		auto result = _lastEntry;
		_lastEntry += size;
		return &_data[result];
	}

	__device__ T* getElementKernel()
	{
		return &_data[_lastEntry++];
	}

	__device__ T* getDataKernel() const
	{
		return _data;
	}
};

double random(double max)
{
	return ((double)rand() / RAND_MAX) * max;
}

__device__ void tiling_Kernel(int numEntities, int division, int numDivisions, int& startIndex, int& endIndex)
{
	int entitiesByDivisions = numEntities / numDivisions;
	int remainder = numEntities % numDivisions;

	int length = division < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
	startIndex = division < remainder ?
		(entitiesByDivisions + 1) * division
		: (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (division - remainder);
	endIndex = startIndex + length - 1;
}

struct CellMapEntry
{
	int clusterIndex;
	int mass;
	double2 cellPos;
	double2 clusterPos;
	double2 clusterVel;
	double clusterAngularVel;
};

struct CellMap
{
	int* map1;
	int* map2;
};

struct Config
{
	int2 size;
};

__device__ void topologyCorrection_Kernel(int2 &pos, Config const &config)
{
	int2 const& size = config.size;
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__device__ void topologyCorrection_Kernel(double2 &pos, Config const &config)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	double2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	topologyCorrection_Kernel(intPart, config);
	pos = { (double)intPart.x + fracPart.x, (double)intPart.y + fracPart.y };
}

__device__ void angleCorrection_Kernel(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__device__ void angleCorrection_Kernel(double &angle)
{
	int intPart = (int)angle;
	double fracPart = angle - intPart;
	angleCorrection_Kernel(intPart);
	angle = (double)intPart + fracPart;
}

__device__ int readCellMapEntry_Kernel(int2 posInt, int clusterIndex, int * __restrict__ map, CellMapEntry* cellMapEntryArray, Config const &config)
{
	topologyCorrection_Kernel(posInt, config);
	auto mapEntry = posInt.x + posInt.y * config.size.x;
	auto slice = config.size.x*config.size.y;

	for (int i = 0; i < LAYERS; ++i) {
		auto index = map[mapEntry + i * slice];
		if (index != -1) {
			if (cellMapEntryArray[index].clusterIndex != clusterIndex) {
				return index;
			}
		}
	}
	return -1;
}

__device__ int readCellMapEntries_Kernel(int2 posInt, int clusterIndex, int * __restrict__ map, ArrayController<CellMapEntry> const& cellMapEntryAC, Config const &config)
{
	--posInt.x;
	--posInt.y;
	CellMapEntry* cellMapEntryArray = cellMapEntryAC.getDataKernel();
	auto index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.x;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.x;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	posInt.x -= 2;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	posInt.x -= 2;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, cellMapEntryArray, config);
	if (index != -1) { return index; }
	return -1;
}

__device__ void movement_Kernel(ClusterCuda* __restrict__ clusters, int clusterIndex, int * __restrict__ oldMap, int * __restrict__ newMap
	, ArrayController<CellMapEntry> const& oldCellMapEntryAC, ArrayController<CellMapEntry> const& newCellMapEntryAC, Config const &config)
{
	ClusterCuda cluster = clusters[clusterIndex];
	if (threadIdx.x >= cluster.numCells) {
		return;
	}

	int startCellIndex;
	int endCellIndex;
	tiling_Kernel(cluster.numCells, threadIdx.x, NUM_THREADS_PER_BLOCK, startCellIndex, endCellIndex);

	__shared__ double rotMatrix[2][2];
	if (threadIdx.x == 0) {
		double sinAngle = __sinf(cluster.angle*DEG_TO_RAD);
		double cosAngle = __cosf(cluster.angle*DEG_TO_RAD);
		rotMatrix[0][0] = cosAngle;
		rotMatrix[0][1] = sinAngle;
		rotMatrix[1][0] = -sinAngle;
		rotMatrix[1][1] = cosAngle;
	}

	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellCuda* cell = &cluster.cells[cellIndex];
		double2 relPos = cell->relPos;
		relPos = { relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1], relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] };
		double2 absPos = { relPos.x + cluster.pos.x, relPos.y + cluster.pos.y };
		int2 absPosInt = { (int)absPos.x , (int)absPos.y };

		readCellMapEntries_Kernel(absPosInt, clusterIndex, oldMap, oldCellMapEntryAC, config);
	}

	if (threadIdx.x == 0) {
		cluster.angle += cluster.angularVel;
		angleCorrection_Kernel(cluster.angle);
		cluster.pos = { cluster.pos.x + cluster.vel.x, cluster.pos.y + cluster.vel.y };
		topologyCorrection_Kernel(cluster.pos, config);
		clusters[clusterIndex] = cluster;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellCuda* cell = &cluster.cells[cellIndex];
		double2 relPos = cell->relPos;
		relPos = { relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1], relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] };
		double2 absPos = { relPos.x + cluster.pos.x, relPos.y + cluster.pos.y };
		int2 absPosInt = { (int)absPos.x , (int)absPos.y };
		topologyCorrection_Kernel(absPosInt, config);
		//		newMap[absPosInt.x + absPosInt.y * config.size.x] = 1;
	}
	__syncthreads();
}


__global__ void movement_Kernel(ClusterCuda* __restrict__ clusters, int * __restrict__ oldMap, int * __restrict__ newMap
	, ArrayController<CellMapEntry> oldCellMapEntryAC, ArrayController<CellMapEntry> newCellMapEntryAC, Config const config)
{
	int blockIndex = blockIdx.x;
	if (blockIndex >= NUM_CLUSTERS) {
		return;
	}

	int startClusterIndex;
	int endClusterIndex;
	tiling_Kernel(NUM_CLUSTERS, blockIndex, NUM_BLOCKS, startClusterIndex, endClusterIndex);

	for (int clusterIndex = startClusterIndex; clusterIndex <= endClusterIndex; ++clusterIndex) {
		movement_Kernel(clusters, clusterIndex, oldMap, newMap, oldCellMapEntryAC, newCellMapEntryAC, config);
	}
}

struct CudaData
{
	cudaStream_t stream;
	Config config;
	ClusterCuda* clusters;
	CellMap map;
	ArrayController<ClusterCuda> clustersAC;
	ArrayController<CellCuda> cellsAC;
	ArrayController<CellMapEntry> cellMapInfoAC1;
	ArrayController<CellMapEntry> cellMapInfoAC2;

} cudaData;

void init_Cuda(int2 size)
{
	cudaStreamCreate(&cudaData.stream);
	cudaSetDevice(0);

	cudaData.config = { size };
	size_t mapSize = size.x * size.y * sizeof(int) * LAYERS;
	cudaMallocManaged(&cudaData.map.map1, mapSize);
	cudaMallocManaged(&cudaData.map.map2, mapSize);
	for (int i = 0; i < size.x * size.y * LAYERS; ++i) {
		cudaData.map.map1[i] = -1;
		cudaData.map.map1[i] = -1;
		cudaData.map.map2[i] = -1;
		cudaData.map.map2[i] = -1;
	}
	int cellsPerCluster = 32;
	cudaData.clustersAC = ArrayController<ClusterCuda>(NUM_CLUSTERS * 2);
	cudaData.cellsAC = ArrayController<CellCuda>(NUM_CLUSTERS * cellsPerCluster * 2);
	cudaData.cellMapInfoAC1 = ArrayController<CellMapEntry>(NUM_CLUSTERS * cellsPerCluster * 2);
	cudaData.cellMapInfoAC2 = ArrayController<CellMapEntry>(NUM_CLUSTERS * cellsPerCluster * 2);

	cudaData.clusters = cudaData.clustersAC.getArray(NUM_CLUSTERS);
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		cudaData.clusters[i].pos = { random(size.x), random(size.y) };
		cudaData.clusters[i].vel = { random(1.0f) - 0.5f, random(1.0) - 0.5f };
		cudaData.clusters[i].angle = random(360.0f);
		cudaData.clusters[i].angularVel = random(10.0f) - 5.0f;
		cudaData.clusters[i].numCells = cellsPerCluster;

		cudaData.clusters[i].cells = cudaData.cellsAC.getArray(cellsPerCluster);
		for (int j = 0; j < cellsPerCluster; ++j) {
			cudaData.clusters[i].cells[j].relPos = { j - 20.0f, j - 20.0f };
		}

	}
}

void calcNextTimestep_Cuda()
{
	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaData.stream>>> (cudaData.clusters, cudaData.map.map1, cudaData.map.map2, cudaData.cellMapInfoAC1
		, cudaData.cellMapInfoAC2, cudaData.config);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
}

void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters)
{
	numClusters = NUM_CLUSTERS;
	clusters = cudaData.clusters;
}


void end_Cuda()
{
	cudaDeviceSynchronize();
	cudaDeviceReset();

	cudaData.cellMapInfoAC2.free();
	cudaData.cellMapInfoAC1.free();
	cudaData.cellsAC.free();
	cudaData.clustersAC.free();
	cudaFree(cudaData.map.map1);
	cudaFree(cudaData.map.map1);
}

/*
int main()
{
	cudaSetDevice(0);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 124);

	Config config{ size };

	CellMap map;
	size_t mapSize = size.x * size.y * sizeof(int) * layers;
	cudaMallocManaged(&map.map1, mapSize);
	cudaMallocManaged(&map.map2, mapSize);
	for (int i = 0; i < size.x * size.y * layers; ++i) {
		map.map1[i] = -1;
		map.map1[i] = -1;
		map.map2[i] = -1;
		map.map2[i] = -1;
	}
	int cellsPerCluster = 32;
	ArrayController<Cluster> clustersAC(numClusters * 2);
	ArrayController<Cell> cellsAC(numClusters * cellsPerCluster * 2);
	ArrayController<CellMapEntry> cellMapInfoAC1(numClusters * cellsPerCluster * 2);
	ArrayController<CellMapEntry> cellMapInfoAC2(numClusters * cellsPerCluster * 2);

	Cluster* clusters = clustersAC.getArray(numClusters);
	for (int i = 0; i < numClusters; ++i) {
		clusters[i].pos = { random(size.x), random(size.y) };
		clusters[i].vel = { random(1.0f) - 0.5f, random(1.0) - 0.5f };
		clusters[i].angle = random(360.0f);
		clusters[i].angularVel = random(10.0f) - 5.0f;
		clusters[i].numCells = cellsPerCluster;

		clusters[i].cells = cellsAC.getArray(cellsPerCluster);
		for (int j = 0; j < cellsPerCluster; ++j) {
			clusters[i].cells[j].relPos = { j - 20.0f, j - 20.0f };
		}

	}

	printf("%f, %f\n", clusters[320].pos.x, clusters[320].vel.x);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int p = 0; p < 1000; ++p) {
		movement_Kernel <<<numBlocks, numThreadsPerBlock >>> (clusters, map.map1, map.map2, cellMapInfoAC1, cellMapInfoAC2, config);
		evaluate();
		cudaDeviceSynchronize();
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed: %f\n", elapsedTime);

	printf(" = %f\n", clusters[320].pos.x);

	cudaDeviceReset();

	cellMapInfoAC2.free();
	cellMapInfoAC1.free();
	cellsAC.free();
	clustersAC.free();
	cudaFree(map.map1);
	cudaFree(map.map1);
	return 0;
}
*/
