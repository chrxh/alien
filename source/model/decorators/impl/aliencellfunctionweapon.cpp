#include "aliencellfunctionweapon.h"

#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"

#include "model/simulationsettings.h"

AlienCellFunctionWeapon::AlienCellFunctionWeapon (AlienCell* cell, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{
}

AlienCellFunctionWeapon::AlienCellFunctionWeapon (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

AlienCellFunctionWeapon::AlienCellFunctionWeapon (AlienCell* cell, QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

AlienCell::ProcessingResult AlienCellFunctionWeapon::process (AlienToken* token, AlienCell* previousCell)
{
    AlienCell::ProcessingResult processingResult = _cell->process(token, previousCell);
    token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::NO_TARGET);
    QVector3D pos = _cell->getCluster()->calcPosition(_cell);
    for(int x = -2; x < 3; ++x)
        for(int y = -2; y < 3; ++y) {
            QVector3D searchPos(pos.x()+x, pos.y()+y, 0.0);
            AlienCell* otherCell = _grid->getCell(searchPos);

            //other cell found?
            if( otherCell ) {
                if( otherCell->getCluster() != _cell->getCluster() ) {
                    qreal energy = otherCell->getEnergy()*simulationParameters.CELL_WEAPON_STRENGTH+1.0;
                    if( otherCell->getEnergy() > energy ) {
                        otherCell->setEnergy(otherCell->getEnergy()-energy);
                        token->energy += energy/2.0;
                        _cell->setEnergy(_cell->getEnergy()+energy/2.0);
                        token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::STRIKE_SUCCESSFUL);
                    }
                }
            }
        }
    return processingResult;
}

