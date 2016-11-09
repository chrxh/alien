#ifndef ALIENENERGYGUIDANCEDECORATORIMPL_H
#define ALIENENERGYGUIDANCEDECORATORIMPL_H

#include "model/decorators/alienenergyguidance.h"

class AlienEnergyGuidanceImpl : public AlienEnergyGuidance
{
public:
    AlienEnergyGuidanceImpl (AlienGrid*& grid);

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
};

#endif // ALIENENERGYGUIDANCEDECORATORIMPL_H
