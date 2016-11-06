#ifndef ALIENENERGYGUIDANCEDECORATORIMPL_H
#define ALIENENERGYGUIDANCEDECORATORIMPL_H

#include "model/decorators/alienenergyguidance.h"

class AlienEnergyGuidanceImpl : public AlienEnergyGuidance
{
public:
    AlienEnergyGuidanceImpl (AlienCell* cell, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell);
};

#endif // ALIENENERGYGUIDANCEDECORATORIMPL_H
