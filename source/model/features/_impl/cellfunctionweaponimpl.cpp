#include "model/simulationunitcontext.h"
#include "model/cellmap.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/simulationparameters.h"
#include "model/modelsettings.h"

#include "cellfunctionweaponimpl.h"

CellFunctionWeaponImpl::CellFunctionWeaponImpl (SimulationUnitContext* context)
    : CellFunction(context)
    , _cellMap(context->getCellMap())
	, _parameters(context->getSimulationParameters())
{
}

CellFeature::ProcessingResult CellFunctionWeaponImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
	auto& tokenMem = token->getMemoryRef();
    tokenMem[Enums::Weapon::OUT] = Enums::WeaponOut::NO_TARGET;
    QVector3D pos = cell->getCluster()->calcPosition(cell);
    for(int x = -2; x < 3; ++x)
        for(int y = -2; y < 3; ++y) {
            QVector3D searchPos(pos.x()+x, pos.y()+y, 0.0);
            Cell* otherCell = _cellMap->getCell(searchPos);

            //other cell found?
            if( otherCell ) {
                if( otherCell->getCluster() != cell->getCluster() ) {
                    qreal energy = otherCell->getEnergy()*_parameters->cellFunctionWeaponStrength+1.0;
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
