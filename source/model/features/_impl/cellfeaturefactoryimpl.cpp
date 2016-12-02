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

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, Grid* grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(grid));
        break;
        case CellFunctionType::PROPULSION :
        registerNewFeature(cell, new CellFunctionPropulsionImpl(grid));
        break;
        case CellFunctionType::SCANNER :
        registerNewFeature(cell, new CellFunctionScannerImpl(grid));
        break;
        case CellFunctionType::WEAPON :
        registerNewFeature(cell, new CellFunctionWeaponImpl(grid));
        break;
        case CellFunctionType::CONSTRUCTOR :
        registerNewFeature(cell, new CellFunctionConstructorImpl(grid));
        break;
        case CellFunctionType::SENSOR :
        registerNewFeature(cell, new CellFunctionSensorImpl(grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(grid));
        break;
        default:
        break;
    }
}

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, quint8* data, Grid* grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(data, grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(data, grid));
        break;
        default:
        addCellFunction(cell, type, grid);
    }
}

void CellDecoratorFactoryImpl::addCellFunction (Cell* cell, CellFunctionType type, QDataStream& stream, Grid* grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new CellFunctionComputerImpl(stream, grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicatorImpl(stream, grid));
        break;
        default:
        addCellFunction(cell, type, grid);
    }
}

void CellDecoratorFactoryImpl::addEnergyGuidance (Cell* cell, Grid* grid)
{
    registerNewFeature(cell, new EnergyGuidanceImpl(grid));
}

CellDecoratorFactoryImpl cellDecoratorFactoryImpl;
