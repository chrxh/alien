#include "cellfeature.h"

CellFeature::~CellFeature ()
{
    delete _nextFeature;
}

void CellFeature::registerNextFeature (CellFeature* nextFeature)
{
    _nextFeature = nextFeature;
}

CellFeature::ProcessingResult CellFeature::process (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult resultFromThisFeature = processImpl(token, cell, previousCell);
    ProcessingResult resultFromNextFeature {false, nullptr };
    if( _nextFeature)
        resultFromNextFeature = _nextFeature->process(token, cell, previousCell);
    ProcessingResult mergedResult;
    mergedResult.decompose = resultFromThisFeature.decompose | resultFromNextFeature.decompose;
    if( resultFromThisFeature.newEnergyParticle )
        mergedResult.newEnergyParticle = resultFromThisFeature.newEnergyParticle;
    else
        mergedResult.newEnergyParticle = resultFromNextFeature.newEnergyParticle;
    return mergedResult;
}

void CellFeature::serializePrimitives (QDataStream& stream) const
{
}

void CellFeature::deserializePrimitives (QDataStream& stream) const
{
}
