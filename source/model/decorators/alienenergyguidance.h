#ifndef ALIENENERGYGUIDANCE_H
#define ALIENENERGYGUIDANCE_H

#include "aliencelldecorator.h"

class AlienEnergyGuidance: public AlienCellDecorator
{
public:
    AlienEnergyGuidance (AlienGrid*& grid) : AlienCellDecorator(grid) {}

    virtual ~AlienEnergyGuidance () {}
};

#endif // ALIENENERGYGUIDANCE_H
