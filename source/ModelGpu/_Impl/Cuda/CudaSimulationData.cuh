#pragma once

struct CudaSimulation
{
	int2 size;

	CudaCell **map1;
	CudaCell **map2;

	ArrayController<CudaCellCluster> clustersAC1;
	ArrayController<CudaCellCluster> clustersAC2;
	ArrayController<CudaCell> cellsAC1;
	ArrayController<CudaCell> cellsAC2;
};
