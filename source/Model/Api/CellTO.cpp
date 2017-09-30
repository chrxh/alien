#include "CellTO.h"

#include "Model/Api/Settings.h"

CellTO::CellTO ()
{
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




