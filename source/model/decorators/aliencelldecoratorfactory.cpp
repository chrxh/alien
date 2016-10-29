#include "aliencellfunctionfactory.h"
#include "aliencellfunctioncomputer.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

AlienCellFunction* AlienCellFunctionFactory::build (Type type, quint8* cellFunctionData, AlienGrid*& grid)
{
    if( type == Type::COMPUTER )
        return new AlienCellFunctionComputer(cellFunctionData, grid);
    if( type == Type::PROPULSION )
        return new AlienCellFunctionPropulsion(cellFunctionData, grid);
    if( type == Type::SCANNER )
        return new AlienCellFunctionScanner(cellFunctionData, grid);
    if( type == Type::WEAPON )
        return new AlienCellFunctionWeapon(cellFunctionData, grid);
    if( type == Type::CONSTRUCTOR )
        return new AlienCellFunctionConstructor(cellFunctionData, grid);
    if( type == Type::SENSOR )
        return new AlienCellFunctionSensor(cellFunctionData, grid);
    if( type == Type::COMMUNICATOR )
        return new AlienCellFunctionCommunicator(cellFunctionData, grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (Type type, AlienGrid*& grid)
{
    if( type == Type::COMPUTER )
        return new AlienCellFunctionComputer(false, grid);
    if( type == Type::PROPULSION )
        return new AlienCellFunctionPropulsion(grid);
    if( type == Type::SCANNER )
        return new AlienCellFunctionScanner(grid);
    if( type == Type::WEAPON )
        return new AlienCellFunctionWeapon(grid);
    if( type == Type::CONSTRUCTOR )
        return new AlienCellFunctionConstructor(grid);
    if( type == Type::SENSOR )
        return new AlienCellFunctionSensor(grid);
    if( type == Type::COMMUNICATOR )
        return new AlienCellFunctionCommunicator(grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QDataStream& stream, AlienGrid*& grid)
{
    QString name;
    stream >> name;
    if( name == "COMPUTER" )
        return new AlienCellFunctionComputer(stream, grid);
    if( name == "PROPULSION" )
        return new AlienCellFunctionPropulsion(stream, grid);
    if( name == "SCANNER" )
        return new AlienCellFunctionScanner(stream, grid);
    if( name == "WEAPON" )
        return new AlienCellFunctionWeapon(stream, grid);
    if( name == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(stream, grid);
    if( name == "SENSOR" )
        return new AlienCellFunctionSensor(stream, grid);
    if( name == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(stream, grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::buildRandomCellFunction (AlienGrid*& grid)
{
    Type type =  static_cast<Type>(qrand()%Type::_COUNTER);
    if( type == Type::COMPUTER )
        return new AlienCellFunctionComputer(true, grid);
    if( type == Type::PROPULSION )
        return build("PROPULSION", grid);
    if( type == Type::SCANNER )
        return build("SCANNER", grid);
    if( type == Type::WEAPON )
        return build("WEAPON", grid);
    if( type == Type::CONSTRUCTOR )
        return build("CONSTRUCTOR", grid);
    if( type == Type::SENSOR )
        return build("SENSOR", grid);
    if( type == Type::COMMUNICATOR )
        return build("COMMUNICATOR", grid);
    return 0;
}

static AlienCellFunctionFactory::Type AlienCellFunctionFactory::getCellFunctionType ()
{

}

