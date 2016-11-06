#include "alienenergyguidanceimpl.h"

#include "model/entities/alientoken.h"
#include "model/decorators/constants.h"
#include "model/simulationsettings.h"


AlienEnergyGuidanceImpl::AlienEnergyGuidanceImpl (AlienCell* cell, AlienGrid*& grid)
    : AlienEnergyGuidance(cell, grid)
{

}

AlienCell::ProcessingResult AlienEnergyGuidanceImpl::process (AlienToken* token, AlienCell* previousCell)
{
    AlienCell::ProcessingResult processingResult = _cell->process(token, previousCell);
    quint8 cmd = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN)] % 6;
    qreal valueCell = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_CELL)];
    qreal valueToken = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_TOKEN)];
    qreal amount = 10.0;

    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_CELL) ) {
        if( _cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            _cell->setEnergy(_cell->getEnergy()-amount);
            token->energy += amount;
        }
        if( _cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
                _cell->setEnergy(_cell->getEnergy()+amount);
                token->energy -= amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            _cell->setEnergy(_cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            if( _cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
                _cell->setEnergy(_cell->getEnergy()-amount);
                token->energy += amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_BOTH) ) {
        if( (token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount))
                && (_cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell)) ) {
            _cell->setEnergy(_cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( (token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken))
                && (_cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount)) ) {
            _cell->setEnergy(_cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_CELL) ) {
        if( _cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
            _cell->setEnergy(_cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
            _cell->setEnergy(_cell->getEnergy()+amount);
            token->energy -= amount;
        }
    }
    return processingResult;
}
