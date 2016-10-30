#ifndef ALIENENERGYGUIDANCE_H
#define ALIENENERGYGUIDANCE_H

#include "aliencelldecorator.h"

class AlienEnergyGuidance: public AlienCellDecorator
{
public:

    AlienEnergyGuidance (AlienCell* cell) : AlienCellDecorator(cell) {}

    virtual ~AlienEnergyGuidance ();

    //constants for cell function programming
    enum class ENERGY_GUIDANCE {
        IN = 1,
        IN_VALUE_CELL = 2,
        IN_VALUE_TOKEN = 3
    };
    enum class ENERGY_GUIDANCE_IN {
        DEACTIVATED,
        BALANCE_CELL,
        BALANCE_TOKEN,
        BALANCE_BOTH,
        HARVEST_CELL,
        HARVEST_TOKEN
    };
};

#endif // ALIENENERGYGUIDANCE_H
