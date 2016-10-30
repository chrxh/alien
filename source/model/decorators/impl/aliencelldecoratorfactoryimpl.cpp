
#include "aliencelldecoratorfactoryimpl.h"

#include "aliencellfunctioncomputerimpl.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (AlienCell* cell, CellFunctionType type, quint8* data, AlienGrid*& grid)
{
    switch( type ) {
        case Type::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, data, grid);
        case Type::PROPULSION :
            return new AlienCellFunctionPropulsion(cell, data, grid);
        case Type::SCANNER :
            return new AlienCellFunctionScanner(cell, data, grid);
        case Type::WEAPON :
            return new AlienCellFunctionWeapon(cell, data, grid);
        case Type::CONSTRUCTOR :
            return new AlienCellFunctionConstructor(cell, data, grid);
        case Type::SENSOR :
            return new AlienCellFunctionSensor(cell, data, grid);
        case Type::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, data, grid);
        default:
            return 0;
    }
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (CellFunctionType type, AlienCell* cell, AlienGrid*& grid)
{
    switch( type ) {
        case Type::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, false, grid);
        case Type::PROPULSION :
            return new AlienCellFunctionPropulsion(cell, grid);
        case Type::SCANNER :
            return new AlienCellFunctionScanner(cell, grid);
        case Type::WEAPON :
            return new AlienCellFunctionWeapon(cell, grid);
        case Type::CONSTRUCTOR :
            return new AlienCellFunctionConstructor(cell, grid);
        case Type::SENSOR :
            return new AlienCellFunctionSensor(cell, grid);
        case Type::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, grid);
        default:
            return 0;
    }
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addCellFunction (QDataStream& stream, AlienGrid*& grid)
{
    QString name;
    stream >> name;
    CellFunctionType type;

    //>>>>>>>>>>>> temp: only for converting
    if( name == "COMPUTER" )
        type = Type::COMPUTER;
    if( name == "PROPULSION" )
        type = Type::PROPULSION;
    if( name == "SCANNER" )
        type = Type::SCANNER;
    if( name == "WEAPON" )
        type = Type::WEAPON;
    if( name == "CONSTRUCTOR" )
        type = Type::CONSTRUCTOR;
    if( name == "SENSOR" )
        type = Type::SENSOR;
    if( name == "COMMUNICATOR" )
        type = Type::COMMUNICATOR;
    //<<<<<<<<<<<<

    switch( type ) {
        case Type::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, stream, grid);
        case Type::PROPULSION :
            return new AlienCellFunctionPropulsion(cell, stream, grid);
        case Type::SCANNER :
            return new AlienCellFunctionScanner(cell, stream, grid);
        case Type::WEAPON :
            return new AlienCellFunctionWeapon(cell, stream, grid);
        case Type::CONSTRUCTOR :
            return new AlienCellFunctionConstructor(cell, stream, grid);
        case Type::SENSOR :
            return new AlienCellFunctionSensor(cell, stream, grid);
        case Type::COMMUNICATOR :
            return new AlienCellFunctionCommunicator(cell, stream, grid);
        default:
            return 0;
    }
}

AlienCellFunction* AlienCellDecoratorFactoryImpl::addRandomCellFunction (AlienCell* cell, AlienGrid*& grid)
{
    CellFunctionType type =  static_cast<CellFunctionType>(qrand()%Type::_COUNTER);

    switch( type ) {
        case Type::COMPUTER :
            return new AlienCellFunctionComputerImpl(cell, true, grid);
        case Type::PROPULSION :
            return build(cell, "PROPULSION", grid);
        case Type::SCANNER :
            return build(cell, "SCANNER", grid);
        case Type::WEAPON :
            return build(cell, "WEAPON", grid);
        case Type::CONSTRUCTOR :
            return build(cell, "CONSTRUCTOR", grid);
        case Type::SENSOR :
            return build(cell, "SENSOR", grid);
        case Type::COMMUNICATOR :
            return build(cell, "COMMUNICATOR", grid);
        default:
            return 0;
    }
}

