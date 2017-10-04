#include "CellFeatureChain.h"

CellFeatureChain::~CellFeatureChain ()
{
    delete _nextFeature;
}

void CellFeatureChain::setContext(UnitContext * context)
{
	_context = context;
	if (_nextFeature) {
		_nextFeature->setContext(context);
	}
}

CellFeatureDescription CellFeatureChain::getDescription() const
{
	CellFeatureDescription result;
	CellFeatureChain const* feature = this;
	while (feature) {
		feature->appendDescriptionImpl(result);
		feature = feature->_nextFeature;
	}
	return result;
}

void CellFeatureChain::registerNextFeature (CellFeatureChain* nextFeature)
{
    _nextFeature = nextFeature;
}

CellFeatureChain::ProcessingResult CellFeatureChain::process (Token* token, Cell* cell, Cell* previousCell)
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

void CellFeatureChain::mutate()
{
	mutateImpl();
	if (_nextFeature) {
		_nextFeature->mutate();
	}
}

