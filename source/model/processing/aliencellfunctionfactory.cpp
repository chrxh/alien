#include "aliencellfunctionfactory.h"
#include "aliencellfunctioncomputer.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

AlienCellFunction* AlienCellFunctionFactory::build (QString name, quint8* cellFunctionData, AlienGrid*& grid)
{
    if( name == "COMPUTER" )
        return new AlienCellFunctionComputer(cellFunctionData, grid);
    if( name == "PROPULSION" )
        return new AlienCellFunctionPropulsion(cellFunctionData, grid);
    if( name == "SCANNER" )
        return new AlienCellFunctionScanner(cellFunctionData, grid);
    if( name == "WEAPON" )
        return new AlienCellFunctionWeapon(cellFunctionData, grid);
    if( name == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(cellFunctionData, grid);
    if( name == "SENSOR" )
        return new AlienCellFunctionSensor(cellFunctionData, grid);
    if( name == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(cellFunctionData, grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QString name, AlienGrid*& grid)
{
    if( name == "COMPUTER" )
        return new AlienCellFunctionComputer(false, grid);
    if( name == "PROPULSION" )
        return new AlienCellFunctionPropulsion(grid);
    if( name == "SCANNER" )
        return new AlienCellFunctionScanner(grid);
    if( name == "WEAPON" )
        return new AlienCellFunctionWeapon(grid);
    if( name == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(grid);
    if( name == "SENSOR" )
        return new AlienCellFunctionSensor(grid);
    if( name == "COMMUNICATOR" )
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
    int name(qrand()%7);
    if( name == 0 )
        return new AlienCellFunctionComputer(true, grid);
    if( name == 1 )
        return build("PROPULSION", grid);
    if( name == 2 )
        return build("SCANNER", grid);
    if( name == 3 )
        return build("WEAPON", grid);
    if( name == 4 )
        return build("CONSTRUCTOR", grid);
    if( name == 5 )
        return build("SENSOR", grid);
    if( name == 6 )
        return build("COMMUNICATOR", grid);
    return 0;
}

