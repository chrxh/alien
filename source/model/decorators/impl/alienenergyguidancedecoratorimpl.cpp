#include "alienenergyguidanceimpl.h"

AlienEnergyGuidanceDecoratorImpl::AlienEnergyGuidanceDecoratorImpl ()
{

}

AlienCell::ProcessingResult AlienEnergyGuidanceDecoratorImpl::process (AlienToken* token, AlienCell* previousCell)
{
    quint8 cmd = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN)] % 6;
    qreal valueCell = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_CELL)];
    qreal valueToken = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_TOKEN)];
    qreal amount = 10.0;

    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_CELL) ) {
        if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
        if( cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
                cell->setEnergy(cell->getEnergy()+amount);
                token->energy -= amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
                cell->setEnergy(cell->getEnergy()-amount);
                token->energy += amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_BOTH) ) {
        if( (token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount))
                && (cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell)) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( (token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken))
                && (cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount)) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_CELL) ) {
        if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
    }
}
