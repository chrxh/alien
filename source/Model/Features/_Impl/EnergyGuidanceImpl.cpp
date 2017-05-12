#include "model/Entities/Token.h"
#include "model/Entities/Cell.h"
#include "model/Features/CellFeatureEnums.h"
#include "model/Settings.h"
#include "model/Context/UnitContext.h"
#include "model/Context/SimulationParameters.h"

#include "EnergyGuidanceImpl.h"


EnergyGuidanceImpl::EnergyGuidanceImpl (UnitContext* context)
    : EnergyGuidance(context)
{

}

CellFeature::ProcessingResult EnergyGuidanceImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
	auto& tokenMem = token->getMemoryRef();
	quint8 cmd = tokenMem[Enums::EnergyGuidance::IN] % 6;
    qreal valueCell = static_cast<quint8>(tokenMem[Enums::EnergyGuidance::IN_VALUE_CELL]);
    qreal valueToken = static_cast<quint8>(tokenMem[Enums::EnergyGuidance::IN_VALUE_TOKEN]);
	auto parameters = _context->getSimulationParameters();
	const qreal amount = 10.0;

    if( cmd == Enums::EnergyGuidanceIn::BALANCE_CELL ) {
        if( cell->getEnergy() > (parameters->cellMinEnergy+valueCell) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->setEnergy(token->getEnergy() + amount);
        }
        if( cell->getEnergy() < (parameters->cellMinEnergy+valueCell) ) {
            if( token->getEnergy() > (parameters->tokenMinEnergy+valueToken+amount) ) {
                cell->setEnergy(cell->getEnergy()+amount);
				token->setEnergy(token->getEnergy() - amount);
            }
        }
    }
    if( cmd == Enums::EnergyGuidanceIn::BALANCE_TOKEN ) {
        if( token->getEnergy() > (parameters->tokenMinEnergy+valueToken) ) {
            cell->setEnergy(cell->getEnergy()+amount);
			token->setEnergy(token->getEnergy() - amount);
		}
        if( token->getEnergy() < (parameters->tokenMinEnergy+valueToken) ) {
            if( cell->getEnergy() > (parameters->cellMinEnergy+valueCell+amount) ) {
                cell->setEnergy(cell->getEnergy()-amount);
				token->setEnergy(token->getEnergy() + amount);
			}
        }
    }
    if( cmd == Enums::EnergyGuidanceIn::BALANCE_BOTH ) {
        if( (token->getEnergy() > (parameters->tokenMinEnergy+valueToken+amount))
                && (cell->getEnergy() < (parameters->cellMinEnergy+valueCell)) ) {
            cell->setEnergy(cell->getEnergy()+amount);
			token->setEnergy(token->getEnergy() - amount);
		}
        if( (token->getEnergy() < (parameters->tokenMinEnergy+valueToken))
                && (cell->getEnergy() > (parameters->cellMinEnergy+valueCell+amount)) ) {
            cell->setEnergy(cell->getEnergy()-amount);
			token->setEnergy(token->getEnergy() + amount);
		}
    }
    if( cmd == Enums::EnergyGuidanceIn::HARVEST_CELL ) {
        if( cell->getEnergy() > (parameters->cellMinEnergy+valueCell+amount) ) {
            cell->setEnergy(cell->getEnergy()-amount);
			token->setEnergy(token->getEnergy() + amount);
		}
    }
    if( cmd == Enums::EnergyGuidanceIn::HARVEST_TOKEN ) {
        if( token->getEnergy() > (parameters->tokenMinEnergy+valueToken+amount) ) {
            cell->setEnergy(cell->getEnergy()+amount);
			token->setEnergy(token->getEnergy() - amount);
		}
    }
    return processingResult;
}
