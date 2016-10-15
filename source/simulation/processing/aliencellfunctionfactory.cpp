#include "aliencellfunctionfactory.h"
#include "aliencellfunctioncomputer.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

AlienCellFunction* AlienCellFunctionFactory::build (QString type, bool randomData, AlienGrid*& grid)
{
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(randomData, grid);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion(grid);
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner(grid);
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon(grid);
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(grid);
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor(grid);
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QDataStream& stream, AlienGrid*& grid)
{
    QString type;
    stream >> type;
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(stream, grid);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion(stream, grid);
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner(stream, grid);
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon(stream, grid);
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(stream, grid);
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor(stream, grid);
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(stream, grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QString type, quint8* cellTypeData, AlienGrid*& grid)
{
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(cellTypeData, grid);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion(cellTypeData, grid);
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner(cellTypeData, grid);
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon(cellTypeData, grid);
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(cellTypeData, grid);
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor(cellTypeData, grid);
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(cellTypeData, grid);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::buildRandom (bool randomData, AlienGrid*& grid)
{
    int type(qrand()%7);
    if( type == 0 )
        return build("COMPUTER", randomData, grid);
    if( type == 1 )
        return build("PROPULSION", randomData, grid);
    if( type == 2 )
        return build("SCANNER", randomData, grid);
    if( type == 3 )
        return build("WEAPON", randomData, grid);
    if( type == 4 )
        return build("CONSTRUCTOR", randomData, grid);
    if( type == 5 )
        return build("SENSOR", randomData, grid);
    if( type == 6 )
        return build("COMMUNICATOR", randomData, grid);
    return 0;
}
