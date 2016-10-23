#include "aliencellfunctionweapon.h"

#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"

#include "model/simulationsettings.h"

AlienCellFunctionWeapon::AlienCellFunctionWeapon(AlienGrid*& grid)
    : AlienCellFunction(grid)
{
}

AlienCellFunctionWeapon::AlienCellFunctionWeapon (quint8* cellFunctionData, AlienGrid*& grid)
    : AlienCellFunction(grid)
{

}

AlienCellFunctionWeapon::AlienCellFunctionWeapon (QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunction(grid)
{

}

void AlienCellFunctionWeapon::execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose)
{
    token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::NO_TARGET);
    QVector3D pos = cell->getCluster()->calcPosition(cell);
    for(int x = -2; x < 3; ++x)
        for(int y = -2; y < 3; ++y) {
            QVector3D searchPos(pos.x()+x, pos.y()+y, 0.0);
            AlienCell* otherCell = _grid->getCell(searchPos);

            //other cell found?
            if( otherCell ) {
                if( otherCell->getCluster() != cell->getCluster() ) {
                    qreal energy = otherCell->getEnergy()*simulationParameters.CELL_WEAPON_STRENGTH+1.0;
                    if( otherCell->getEnergy() > energy ) {
                        otherCell->setEnergy(otherCell->getEnergy()-energy);
                        token->energy += energy/2.0;
                        cell->setEnergy(cell->getEnergy()+energy/2.0);
                        token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::STRIKE_SUCCESSFUL);
                    }
                }
            }
        }
}

QString AlienCellFunctionWeapon::getCellFunctionName () const
{
    return "WEAPON";
}

void AlienCellFunctionWeapon::serialize (QDataStream& stream)
{
    AlienCellFunction::serialize(stream);
}
