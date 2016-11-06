#ifndef ALIENENERGYGUIDANCE_H
#define ALIENENERGYGUIDANCE_H

#include "aliencelldecorator.h"

class AlienEnergyGuidance: public AlienCellDecorator
{
public:
    AlienEnergyGuidance (AlienCell* cell, AlienGrid*& grid) : AlienCellDecorator(cell, grid) {}

    virtual ~AlienEnergyGuidance () {}
};

#endif // ALIENENERGYGUIDANCE_H
