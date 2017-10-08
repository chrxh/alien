#include "CellFeatureFactoryImpl.h"

#include "Model/Local/Cell.h"
#include "CellComputerFunctionImpl.h"
#include "ConstructorFunction.h"
#include "PropulsionFunction.h"
#include "ScannerFunction.h"
#include "WeaponFunction.h"
#include "SensorFunction.h"
#include "CommunicatorFunction.h"
#include "EnergyGuidanceImpl.h"

namespace {
    CellFeatureChain* registerNewFeature (Cell* cell, CellFeatureChain* newFeature)
    {
        CellFeatureChain* features = cell->getFeatures();
        if( features ) {
            features->registerNextFeature(newFeature);
        }
        else
            cell->registerFeatures(newFeature);
        return newFeature;
    }
}

CellFeatureChain* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellComputerFunctionImpl(context));
        case Enums::CellFunction::PROPULSION :
            return registerNewFeature(cell, new PropulsionFunction(context));
        case Enums::CellFunction::SCANNER :
            return registerNewFeature(cell, new ScannerFunction(context));
        case Enums::CellFunction::WEAPON :
            return registerNewFeature(cell, new WeaponFunction(context));
        case Enums::CellFunction::CONSTRUCTOR :
            return registerNewFeature(cell, new ConstructorFunction(context));
        case Enums::CellFunction::SENSOR :
            return registerNewFeature(cell, new SensorFunction(context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CommunicatorFunction(context));
        default:
            return nullptr;
    }
}

CellFeatureChain* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data
    , UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellComputerFunctionImpl(data, context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CommunicatorFunction(data, context));
        default:
            return addCellFunction(cell, type, context);
    }
}

CellFeatureChain* CellFeatureFactoryImpl::addEnergyGuidance (Cell* cell, UnitContext* context) const
{
    return registerNewFeature(cell, new EnergyGuidanceImpl(context));
}

