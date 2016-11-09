#include "aliencelldecorator.h"

AlienCellDecorator::~AlienCellDecorator ()
{
    if( _nextFeature )
        delete _nextFeature;
}

void AlienCellDecorator::registerNextFeature (AlienCellDecorator* nextFeature)
{
    _nextFeature = nextFeature;
}

AlienCellDecorator::ProcessingResult AlienCellDecorator::process (AlienToken* token, AlienCell* cell, AlienCell* previousCell)
{
    ProcessingResult resultFromNextFeature {false, 0 };
    if( _nextFeature)
        resultFromNextFeature = _nextFeature->process(token, cell, previousCell);
    ProcessingResult resultFromThisFeature = processImpl(token, cell, previousCell);
    ProcessingResult mergedResult;
    mergedResult.decompose = resultFromThisFeature.decompose | resultFromNextFeature.decompose;
    if( resultFromThisFeature.newEnergyParticle )
        mergedResult.newEnergyParticle = resultFromThisFeature.newEnergyParticle;
    else
        mergedResult.newEnergyParticle = resultFromNextFeature.newEnergyParticle;
    return mergedResult;
}

void AlienCellDecorator::serialize (QDataStream& stream) const
{
    serializeImpl(stream);
    if( _nextFeature)
        _nextFeature->serialize(stream);
}

