#pragma once

struct CudaData
{
	int2 size;

	CellCuda **map1;
	CellCuda **map2;

	ArrayController<ClusterCuda> clustersAC1;
	ArrayController<ClusterCuda> clustersAC2;
	ArrayController<CellCuda> cellsAC1;
	ArrayController<CellCuda> cellsAC2;

};
