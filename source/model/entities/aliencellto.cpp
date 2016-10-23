#include "aliencellto.h"

#include "aliencell.h"
#include "aliencellcluster.h"
#include "model/processing/aliencellfunction.h"

#include "global/simulationsettings.h"

AlienCellTO::AlienCellTO ()
    : numCells(0), clusterPos(0.0, 0.0, 0.0), clusterVel(0.0, 0.0, 0.0),
      clusterAngle(0.0), clusterAngVel(0.0), cellPos(0.0, 0.0, 0.0),
      cellEnergy(0.0), cellNumCon(0), cellMaxCon(0), cellAllowToken(true),
      cellTokenAccessNum(0), computerMemory(simulationParameters.CELL_MEMSIZE, 0)
{
/*    for(int i = 0; i < CELL_MEMSIZE; ++i)
        computerMemory[i] = 0;*/
}

AlienCellTO::AlienCellTO (AlienCell* cell)
    : computerMemory(simulationParameters.CELL_MEMSIZE)
{

    //copy cell properties
    AlienCellCluster* cluster = cell->getCluster();
    numCells = cluster->getMass();
    clusterPos = cluster->getPosition();
    clusterVel = cluster->getVel();
    clusterAngle = cluster->getAngle();
    clusterAngVel = cluster->getAngularVel();
    cellPos = cell->calcPosition();
    cellEnergy = cell->getEnergy();
    cellNumCon = cell->getNumConnections();
    cellMaxCon = cell->getMaxConnections();
    cellAllowToken = !cell->blockToken();
    cellTokenAccessNum = cell->getTokenAccessNumber();
    cellFunctionName = cell->getCellFunction()->getCellFunctionName();

    //copy computer data
    QVector< quint8 > d = cell->getMemory();
    for(int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i)
        computerMemory[i] = d[i];
    computerCode = cell->getCellFunction()->getCode();

    //copy token data
    for(int i = 0; i < cell->getNumToken(); ++i) {
        AlienToken* token = cell->getToken(i);
        tokenEnergies << token->energy;
        QVector< quint8 > d(simulationParameters.TOKEN_MEMSIZE);
        for(int j = 0; j < simulationParameters.TOKEN_MEMSIZE; ++j)
            d[j] = token->memory[j];
        tokenData << d;
    }
}

AlienCellTO::~AlienCellTO ()
{
}

void AlienCellTO::copyCellProperties (const AlienCellTO& otherCell)
{
    cellPos = otherCell.cellPos;
    cellEnergy = otherCell.cellEnergy;
    cellMaxCon = otherCell.cellMaxCon;
    cellAllowToken = otherCell.cellAllowToken;
    cellTokenAccessNum = otherCell.cellTokenAccessNum;
    cellFunctionName = otherCell.cellFunctionName;
}

void AlienCellTO::copyClusterProperties (const AlienCellTO& otherCell)
{
    clusterPos = otherCell.clusterPos;
    clusterVel = otherCell.clusterVel;
    clusterAngle = otherCell.clusterAngle;
    clusterAngVel = otherCell.clusterAngVel;
}




