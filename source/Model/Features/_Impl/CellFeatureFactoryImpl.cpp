#include "CellFeatureFactoryImpl.h"

#include "model/Entities/Cell.h"
#include "CellComputerImpl.h"
#include "CellConstructorImpl.h"
#include "CellPropulsionImpl.h"
#include "CellScannerImpl.h"
#include "CellWeaponImpl.h"
#include "CellSensorImpl.h"
#include "CellCommunicatorImpl.h"
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
            return registerNewFeature(cell, new CellComputerImpl(context));
        case Enums::CellFunction::PROPULSION :
            return registerNewFeature(cell, new CellPropulsionImpl(context));
        case Enums::CellFunction::SCANNER :
            return registerNewFeature(cell, new CellScannerImpl(context));
        case Enums::CellFunction::WEAPON :
            return registerNewFeature(cell, new CellWeaponImpl(context));
        case Enums::CellFunction::CONSTRUCTOR :
            return registerNewFeature(cell, new CellConstructorImpl(context));
        case Enums::CellFunction::SENSOR :
            return registerNewFeature(cell, new CellSensorImpl(context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CellCommunicatorImpl(context));
        default:
            return nullptr;
    }
}

CellFeature* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, QByteArray data
    , UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellComputerImpl(data, context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CellCommunicatorImpl(data, context));
        default:
            return addCellFunction(cell, type, context);
    }
}

CellFeature* CellFeatureFactoryImpl::addEnergyGuidance (Cell* cell, UnitContext* context) const
{
    return registerNewFeature(cell, new EnergyGuidanceImpl(context));
}

