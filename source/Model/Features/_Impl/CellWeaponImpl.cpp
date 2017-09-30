#include "Model/Context/UnitContext.h"
#include "Model/Context/CellMap.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Token.h"
#include "Model/SimulationParameters.h"
#include "Model/Settings.h"

#include "CellWeaponImpl.h"

CellWeaponImpl::CellWeaponImpl (UnitContext* context)
    : CellFunction(context)
{
}

CellFeature::ProcessingResult CellWeaponImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
	auto cellMap = _context->getCellMap();
	auto parameters = _context->getSimulationParameters();

	auto& tokenMem = token->getMemoryRef();
    tokenMem[Enums::Weapon::OUT] = Enums::WeaponOut::NO_TARGET;
    QVector2D pos = cell->getCluster()->calcPosition(cell);
    for(int x = -2; x < 3; ++x)
        for(int y = -2; y < 3; ++y) {
            QVector2D searchPos(pos.x()+x, pos.y()+y);
            Cell* otherCell = cellMap->getCell(searchPos);

            //other cell found?
            if( otherCell ) {
                if( otherCell->getCluster() != cell->getCluster() ) {
                    qreal energy = otherCell->getEnergy()*parameters->cellFunctionWeaponStrength+1.0;
                    if( otherCell->getEnergy() > energy ) {
                        otherCell->setEnergy(otherCell->getEnergy()-energy);
                        token->setEnergy(token->getEnergy() + energy/2.0);
                        cell->setEnergy(cell->getEnergy()+energy/2.0);
                        tokenMem[Enums::Weapon::OUT] = Enums::WeaponOut::STRIKE_SUCCESSFUL;
                    }
                }
            }
        }
    return processingResult;
}
