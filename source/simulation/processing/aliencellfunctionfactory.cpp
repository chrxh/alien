#include "aliencellfunctionfactory.h"
#include "aliencellfunctioncomputer.h"
#include "aliencellfunctionconstructor.h"
#include "aliencellfunctionpropulsion.h"
#include "aliencellfunctionscanner.h"
#include "aliencellfunctionweapon.h"
#include "aliencellfunctionsensor.h"
#include "aliencellfunctioncommunicator.h"

AlienCellFunction* AlienCellFunctionFactory::build (QString type, bool randomData)
{
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(randomData);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion();
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner();
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon();
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor();
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor();
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator();
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QDataStream& stream)
{
    QString type;
    stream >> type;
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(stream);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion(stream);
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner(stream);
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon(stream);
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(stream);
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor(stream);
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(stream);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::build (QString type, quint8* cellTypeData)
{
    if( type == "COMPUTER" )
        return new AlienCellFunctionComputer(cellTypeData);
    if( type == "PROPULSION" )
        return new AlienCellFunctionPropulsion(cellTypeData);
    if( type == "SCANNER" )
        return new AlienCellFunctionScanner(cellTypeData);
    if( type == "WEAPON" )
        return new AlienCellFunctionWeapon(cellTypeData);
    if( type == "CONSTRUCTOR" )
        return new AlienCellFunctionConstructor(cellTypeData);
    if( type == "SENSOR" )
        return new AlienCellFunctionSensor(cellTypeData);
    if( type == "COMMUNICATOR" )
        return new AlienCellFunctionCommunicator(cellTypeData);
    return 0;
}

AlienCellFunction* AlienCellFunctionFactory::buildRandom (bool randomData)
{
    int type(qrand()%7);
    if( type == 0 )
        return build("COMPUTER", randomData);
    if( type == 1 )
        return build("PROPULSION", randomData);
    if( type == 2 )
        return build("SCANNER", randomData);
    if( type == 3 )
        return build("WEAPON", randomData);
    if( type == 4 )
        return build("CONSTRUCTOR", randomData);
    if( type == 5 )
        return build("SENSOR", randomData);
    if( type == 6 )
        return build("COMMUNICATOR", randomData);
    return 0;
}

/*int AlienCellFunctionFactory::convertFunctionNameToCellType (QString name)
{
    if( name == "COMPUTER" )
        return 0;
    if( name == "PROPULSION" )
        return 1;
    if( name == "SCANNER" )
        return 2;
    if( name == "WEAPON" )
        return 3;
    if( name == "CONSTRUCTOR" )
        return 4;
    return 0;
}*/
