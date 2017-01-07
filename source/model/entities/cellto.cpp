#include "cellto.h"

#include "model/config.h"

CellTO::CellTO ()
    : numCells(0), clusterPos(0.0, 0.0, 0.0), clusterVel(0.0, 0.0, 0.0),
      clusterAngle(0.0), clusterAngVel(0.0), cellPos(0.0, 0.0, 0.0),
      cellEnergy(0.0), cellNumCon(0), cellMaxCon(0), cellAllowToken(true),
      cellTokenAccessNum(0), computerMemory(simulationParameters.CELL_MEMSIZE, 0)
{
/*    for(int i = 0; i < CELL_MEMSIZE; ++i)
        computerMemory[i] = 0;*/
}

CellTO::~CellTO ()
{
}

void CellTO::copyCellProperties (const CellTO& otherCell)
{
    cellPos = otherCell.cellPos;
    cellEnergy = otherCell.cellEnergy;
    cellMaxCon = otherCell.cellMaxCon;
    cellAllowToken = otherCell.cellAllowToken;
    cellTokenAccessNum = otherCell.cellTokenAccessNum;
    cellFunctionType = otherCell.cellFunctionType;
}

void CellTO::copyClusterProperties (const CellTO& otherCell)
{
    clusterPos = otherCell.clusterPos;
    clusterVel = otherCell.clusterVel;
    clusterAngle = otherCell.clusterAngle;
    clusterAngVel = otherCell.clusterAngVel;
}




