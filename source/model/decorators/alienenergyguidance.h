#ifndef ALIENENERGYGUIDANCE_H
#define ALIENENERGYGUIDANCE_H

#include "aliencelldecorator.h"

class AlienEnergyGuidance: public AlienCellDecorator
{
public:

    AlienEnergyGuidance (AlienCell* cell) : AlienCellDecorator(cell) {}

    virtual ~AlienEnergyGuidance ();
};

#endif // ALIENENERGYGUIDANCE_H
