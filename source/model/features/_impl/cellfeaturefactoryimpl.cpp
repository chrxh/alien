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
    void registerNewFeature (Cell* cell, CellFeature* newFeature)
    {
        CellFeature* features = cell->getFeatures();
        if( features ) {
            features->registerNextFeature(newFeature);
        }
        else
            cell->registerFeatures(newFeature);
    }
}

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, SimulationContext* context) const
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(contex));
        break;
        case CellFunctionType::PROPULSION :
        registerNewFeature(cell, new CellFunctionPropulsionImpl(context));
        break;
        case CellFunctionType::SCANNER :
        registerNewFeature(cell, new CellFunctionScannerImpl(context));
        break;
        case CellFunctionType::WEAPON :
        registerNewFeature(cell, new CellFunctionWeaponImpl(context));
        break;
        case CellFunctionType::CONSTRUCTOR :
        registerNewFeature(cell, new CellFunctionConstructorImpl(context));
        break;
        case CellFunctionType::SENSOR :
        registerNewFeature(cell, new CellFunctionSensorImpl(context));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(context));
        break;
        default:
        break;
    }
}

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, quint8* data
    , SimulationContext* context) const
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(data, context));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(data, context));
        break;
        default:
        addCellFunction(cell, type, context);
    }
}

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, QDataStream& stream
    , SimulationContext* context) const
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(stream, context));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(stream, context));
        break;
        default:
        addCellFunction(cell, type, context);
    }
}

void CellDecoratorFactoryImpl::addEnergyGuidance (Cell* cell, SimulationContext* context) const
{
    registerNewFeature(cell, new EnergyGuidanceImpl(context));
}

CellDecoratorFactoryImpl cellDecoratorFactoryImpl;
