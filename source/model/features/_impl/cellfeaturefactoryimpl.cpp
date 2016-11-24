#include "cellfeaturefactoryimpl.h"

#include "model/entities/cell.h"
#include "cellfunctioncomputerimpl.h"
#include "cellfunctionconstructor.h"
#include "cellfunctionpropulsion.h"
#include "cellfunctionscanner.h"
#include "cellfunctionweapon.h"
#include "cellfunctionsensor.h"
#include "cellfunctioncommunicator.h"
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
        registerNewFeature(cell, new CellFunctionPropulsion(grid));
        break;
        case CellFunctionType::SCANNER :
        registerNewFeature(cell, new CellFunctionScanner(grid));
        break;
        case CellFunctionType::WEAPON :
        registerNewFeature(cell, new CellFunctionWeapon(grid));
        break;
        case CellFunctionType::CONSTRUCTOR :
        registerNewFeature(cell, new CellFunctionConstructor(grid));
        break;
        case CellFunctionType::SENSOR :
        registerNewFeature(cell, new CellFunctionSensor(grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new CellFunctionCommunicator(grid));
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
        registerNewFeature(cell, new CellFunctionCommunicator(data, grid));
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
        registerNewFeature(cell, new CellFunctionCommunicator(stream, grid));
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
