#include "cellfeaturefactoryimpl.h"

#include "model/entities/cell.h"
#include "cellfunctioncomputerimpl.h"
#include "cellfunctionconstructorimpl.h"
#include "cellfunctionpropulsionimpl.h"
#include "cellfunctionscannerimpl.h"
#include "cellfunctionweaponimpl.h"
#include "cellfunctionsensorimpl.h"
#include "cellfunctioncommunicatorimpl.h"
#include "energyguidanceimpl.h"

#include "global/servicelocator.h"

CellDecoratorFactoryImpl::CellDecoratorFactoryImpl ()
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

CellFeature* CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, SimulationContext* context) const
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
            return registerNewFeature(cell, new CellFunctionComputerImpl(contex));
        case CellFunctionType::PROPULSION :
            return registerNewFeature(cell, new CellFunctionPropulsionImpl(context));
        case CellFunctionType::SCANNER :
            return registerNewFeature(cell, new CellFunctionScannerImpl(context));
        case CellFunctionType::WEAPON :
            return registerNewFeature(cell, new CellFunctionWeaponImpl(context));
        case CellFunctionType::CONSTRUCTOR :
            return registerNewFeature(cell, new CellFunctionConstructorImpl(context));
        case CellFunctionType::SENSOR :
            return registerNewFeature(cell, new CellFunctionSensorImpl(context));
        case CellFunctionType::COMMUNICATOR :
            return registerNewFeature(cell, new CellFunctionCommunicatorImpl(context));
        default:
            return nullptr;
    }
}

CellFeature* CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, quint8* data
    , SimulationContext* context) const
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
            return registerNewFeature(cell, new CellFunctionComputerImpl(data, context));
        case CellFunctionType::COMMUNICATOR :
            return registerNewFeature(cell, new CellFunctionCommunicatorImpl(data, context));
        default:
            return addCellFunction(cell, type, context);
    }
}

CellFeature* CellDecoratorFactoryImpl::addEnergyGuidance (Cell* cell, SimulationContext* context) const
{
    return registerNewFeature(cell, new EnergyGuidanceImpl(context));
}

CellDecoratorFactoryImpl cellDecoratorFactoryImpl;
