#ifndef ALIENENERGYGUIDANCEDECORATOR_H
#define ALIENENERGYGUIDANCEDECORATOR_H

#include "model/entities/aliencell.h"

class AlienEnergyGuidanceDecorator : public AlienCell
{
public:

    AlienEnergyGuidanceDecorator (AlienCell* cell) : _cell(cell) {}

    virtual ~AlienEnergyGuidanceDecorator ();

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

protected:
    AlienCell* _cell;

};

#endif // ALIENENERGYGUIDANCEDECORATOR_H
