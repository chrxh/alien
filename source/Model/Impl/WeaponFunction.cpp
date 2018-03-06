#include "Model/Local/UnitContext.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/Cell.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Token.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/Settings.h"

#include "WeaponFunction.h"

WeaponFunction::WeaponFunction (UnitContext* context)
    : CellFunction(context)
{
}

CellFeatureChain::ProcessingResult WeaponFunction::processImpl (Token* token, Cell* cell, Cell* previousCell)
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
