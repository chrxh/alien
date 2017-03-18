#include "model/simulationcontext.h"
#include "model/cellmap.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/simulationparameters.h"
#include "model/config.h"

#include "cellfunctionweaponimpl.h"

CellFunctionWeaponImpl::CellFunctionWeaponImpl (SimulationContext* context)
    : CellFunction(context)
    , _cellMap(context->getCellMap())
	, _parameters(context->getSimulationParameters())
{
}

CellFeature::ProcessingResult CellFunctionWeaponImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::NO_TARGET);
    QVector3D pos = cell->getCluster()->calcPosition(cell);
    for(int x = -2; x < 3; ++x)
        for(int y = -2; y < 3; ++y) {
            QVector3D searchPos(pos.x()+x, pos.y()+y, 0.0);
            Cell* otherCell = _cellMap->getCell(searchPos);

            //other cell found?
            if( otherCell ) {
                if( otherCell->getCluster() != cell->getCluster() ) {
                    qreal energy = otherCell->getEnergy()*_parameters->CELL_WEAPON_STRENGTH+1.0;
                    if( otherCell->getEnergy() > energy ) {
                        otherCell->setEnergy(otherCell->getEnergy()-energy);
                        token->energy += energy/2.0;
                        cell->setEnergy(cell->getEnergy()+energy/2.0);
                        token->memory[static_cast<int>(WEAPON::OUT)] = static_cast<int>(WEAPON_OUT::STRIKE_SUCCESSFUL);
                    }
                }
            }
        }
    return processingResult;
}
