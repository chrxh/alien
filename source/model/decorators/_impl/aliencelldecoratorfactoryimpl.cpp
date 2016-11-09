#include "aliencelldecoratorfactoryimpl.h"

#include "model/entities/aliencell.h"
#include "aliencellfunctioncomputerimpl.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"
#include "alienenergyguidanceimpl.h"

#include "global/servicelocator.h"

AlienCellDecoratorFactoryImpl::AlienCellDecoratorFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<AlienCellDecoratorFactory>(this);
}

namespace {
    void registerNewFeature (AlienCell* cell, AlienCellDecorator* newFeature)
    {
        AlienCellDecorator* features = cell->getFeatureChain();
        if( features )
            features->registerNextFeature(newFeature);
        else
            cell->registerFeatureChain(newFeature);
    }
}

void AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new AlienCellFunctionComputerImpl(grid));
        break;
        case CellFunctionType::PROPULSION :
        registerNewFeature(cell, new AlienCellFunctionPropulsion(grid));
        break;
        case CellFunctionType::SCANNER :
        registerNewFeature(cell, new AlienCellFunctionScanner(grid));
        break;
        case CellFunctionType::WEAPON :
        registerNewFeature(cell, new AlienCellFunctionWeapon(grid));
        break;
        case CellFunctionType::CONSTRUCTOR :
        registerNewFeature(cell, new AlienCellFunctionConstructor(grid));
        break;
        case CellFunctionType::SENSOR :
        registerNewFeature(cell, new AlienCellFunctionSensor(grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new AlienCellFunctionCommunicator(grid));
        break;
        default:
        break;
    }
}

void AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new AlienCellFunctionComputerImpl(data, grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new AlienCellFunctionCommunicator(data, grid));
        break;
        default:
        addCellFunction(cell, type, grid);
    }
}

void AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, QDataStream& stream, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
        registerNewFeature(cell, new AlienCellFunctionComputerImpl(stream, grid));
        break;
        case CellFunctionType::COMMUNICATOR :
        registerNewFeature(cell, new AlienCellFunctionCommunicator(stream, grid));
        break;
        default:
        addCellFunction(cell, type, grid);
    }
}

void AlienCellDecoratorFactoryImpl::addEnergyGuidance (AlienCell* cell, AlienGrid*& grid)
{
    registerNewFeature(cell, new AlienEnergyGuidanceImpl(grid));
}

AlienCellDecoratorFactoryImpl cellDecoratorFactoryImpl;
