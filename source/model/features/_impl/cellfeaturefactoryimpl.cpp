#include "CellFeatureFactoryImpl.h"

#include "model/entities/Cell.h"
#include "CellFunctionComputerImpl.h"
#include "CellFunctionConstructorImpl.h"
#include "CellFunctionPropulsionImpl.h"
#include "CellFunctionScannerImpl.h"
#include "CellFunctionWeaponImpl.h"
#include "CellFunctionSensorImpl.h"
#include "CellFunctionCommunicatorImpl.h"
#include "EnergyGuidanceImpl.h"

#include "Base/ServiceLocator.h"

namespace {
	CellFeatureFactoryImpl cellDecoratorFactoryImpl;
}

CellFeatureFactoryImpl::CellFeatureFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<CellFeatureFactory>(this);
}

namespace {
    CellFeature* registerNewFeature (Cell* cell, CellFeature* newFeature)
    {
        CellFeature* features = cell->getFeatures();
        if( features ) {
            features->registerNextFeature(newFeature);
        }
        else
            cell->registerFeatures(newFeature);
        return newFeature;
    }
}

CellFeature* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellFunctionComputerImpl(context));
        case Enums::CellFunction::PROPULSION :
            return registerNewFeature(cell, new CellFunctionPropulsionImpl(context));
        case Enums::CellFunction::SCANNER :
            return registerNewFeature(cell, new CellFunctionScannerImpl(context));
        case Enums::CellFunction::WEAPON :
            return registerNewFeature(cell, new CellFunctionWeaponImpl(context));
        case Enums::CellFunction::CONSTRUCTOR :
            return registerNewFeature(cell, new CellFunctionConstructorImpl(context));
        case Enums::CellFunction::SENSOR :
            return registerNewFeature(cell, new CellFunctionSensorImpl(context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CellFunctionCommunicatorImpl(context));
        default:
            return nullptr;
    }
}

CellFeature* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data
    , UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellFunctionComputerImpl(data, context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CellFunctionCommunicatorImpl(data, context));
        default:
            return addCellFunction(cell, type, context);
    }
}

CellFeature* CellFeatureFactoryImpl::addEnergyGuidance (Cell* cell, UnitContext* context) const
{
    return registerNewFeature(cell, new EnergyGuidanceImpl(context));
}

