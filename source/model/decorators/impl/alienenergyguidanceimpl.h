#ifndef ALIENENERGYGUIDANCEDECORATORIMPL_H
#define ALIENENERGYGUIDANCEDECORATORIMPL_H

#include "model/decorators/alienenergyguidance.h"

class AlienEnergyGuidanceDecoratorImpl : public AlienEnergyGuidanceDecorator
{
public:
    AlienEnergyGuidanceDecoratorImpl ();
    ~AlienEnergyGuidanceDecoratorImpl () {}

    ProcessingResult process (AlienToken* token, AlienCell* previousCell);
};

#endif // ALIENENERGYGUIDANCEDECORATORIMPL_H
