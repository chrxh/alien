#include "Token.h"
#include "Cell.h"
#include "ModelBasic/CellFeatureEnums.h"
#include "ModelBasic/Settings.h"
#include "UnitContext.h"
#include "ModelBasic/SimulationParameters.h"

#include "EnergyGuidanceImpl.h"


EnergyGuidanceImpl::EnergyGuidanceImpl (UnitContext* context)
    : EnergyGuidance(context)
{

}

CellFeatureChain::ProcessingResult EnergyGuidanceImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
	auto& tokenMem = token->getMemoryRef();
	quint8 cmd = tokenMem[Enums::EnergyGuidance::IN] % 6;
    qreal valueCell = static_cast<quint8>(tokenMem[Enums::EnergyGuidance::IN_VALUE_CELL]);
    qreal valueToken = static_cast<quint8>(tokenMem[Enums::EnergyGuidance::IN_VALUE_TOKEN]);
	auto parameters = _context->getSimulationParameters();
	const qreal amount = 10.0;

    if (cmd == Enums::EnergyGuidanceIn::BALANCE_CELL ) {
        if (cell->getEnergy() > (parameters->cellMinEnergy + valueCell + amount)) {
			cell->setEnergy(cell->getEnergy() - amount);
			token->setEnergy(token->getEnergy() + amount);
        }
		else if(token->getEnergy() > (parameters->tokenMinEnergy + valueToken + amount)) {
            cell->setEnergy(cell->getEnergy()+amount);
			token->setEnergy(token->getEnergy() - amount);
        }
    }
    if (cmd == Enums::EnergyGuidanceIn::BALANCE_TOKEN ) {
        if (token->getEnergy() > (parameters->tokenMinEnergy + valueToken + amount)) {
			cell->setEnergy(cell->getEnergy() + amount);
			token->setEnergy(token->getEnergy() - amount);
		}
		else if (cell->getEnergy() > (parameters->cellMinEnergy + valueCell + amount)) {
            cell->setEnergy(cell->getEnergy()-amount);
			token->setEnergy(token->getEnergy() + amount);
        }
    }
    if( cmd == Enums::EnergyGuidanceIn::BALANCE_BOTH ) {
        if( (token->getEnergy() > (parameters->tokenMinEnergy + valueToken + amount))
                && (cell->getEnergy() < (parameters->cellMinEnergy + valueCell)) ) {
            cell->setEnergy(cell->getEnergy() + amount);
			token->setEnergy(token->getEnergy() - amount);
			return processingResult;
		}
        if( (token->getEnergy() < (parameters->tokenMinEnergy + valueToken))
                && (cell->getEnergy() > (parameters->cellMinEnergy + valueCell + amount)) ) {
            cell->setEnergy(cell->getEnergy() - amount);
			token->setEnergy(token->getEnergy() + amount);
			return processingResult;
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
