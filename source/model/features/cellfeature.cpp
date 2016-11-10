#include "cellfeature.h"

CellDecorator::~CellDecorator ()
{
    if( _nextFeature )
        delete _nextFeature;
}

void CellDecorator::registerNextFeature (CellDecorator* nextFeature)
{
    _nextFeature = nextFeature;
}

CellDecorator::ProcessingResult CellDecorator::process (Token* token, Cell* cell, Cell* previousCell)
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

void CellDecorator::serialize (QDataStream& stream) const
{
    serializeImpl(stream);
    if( _nextFeature)
        _nextFeature->serialize(stream);
}

