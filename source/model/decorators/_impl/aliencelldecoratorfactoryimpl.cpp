#include "aliencelldecoratorfactoryimpl.h"

#include "aliencellfunctioncomputerimpl.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

#include "global/servicelocator.h"

AlienCellDecoratorFactoryImpl::AlienCellDecoratorFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<AlienCellDecoratorFactory>(this);
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, grid);
        case CellFunctionType::PROPULSION :
            return new AlienCellFunctionPropulsion(cell, grid);
        case CellFunctionType::SCANNER :
            return new AlienCellFunctionScanner(cell, grid);
        case CellFunctionType::WEAPON :
            return new AlienCellFunctionWeapon(cell, grid);
        case CellFunctionType::CONSTRUCTOR :
            return new AlienCellFunctionConstructor(cell, grid);
        case CellFunctionType::SENSOR :
            return new AlienCellFunctionSensor(cell, grid);
        case CellFunctionType::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, grid);
        default:
            return 0;
    }
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, data, grid);
        case CellFunctionType::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, data, grid);
        default:
            return addCellFunction(cell, type, grid);
    }
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, QDataStream& stream, AlienGrid*& grid)
{
    switch( type ) {
        case CellFunctionType::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, stream, grid);
        case CellFunctionType::PROPULSION :
            return new AlienCellFunctionPropulsion(cell, stream, grid);
        case CellFunctionType::SCANNER :
            return new AlienCellFunctionScanner(cell, stream, grid);
        case CellFunctionType::WEAPON :
            return new AlienCellFunctionWeapon(cell, stream, grid);
        case CellFunctionType::CONSTRUCTOR :
            return new AlienCellFunctionConstructor(cell, stream, grid);
        case CellFunctionType::SENSOR :
            return new AlienCellFunctionSensor(cell, stream, grid);
        case CellFunctionType::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, stream, grid);
        default:
            return 0;
    }
}

AlienCellDecoratorFactoryImpl cellDecoratorFactoryImpl;
