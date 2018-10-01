#include "CellFeatureFactoryImpl.h"

#include "CellComputerFunctionImpl.h"
#include "ConstructorFunction.h"
#include "PropulsionFunction.h"
#include "ScannerFunction.h"
#include "WeaponFunction.h"
#include "SensorFunction.h"
#include "CommunicatorFunction.h"
#include "EnergyGuidanceImpl.h"
#include "Cell.h"

namespace
{
	Enums::CellFunction::Type modulo(Enums::CellFunction::Type value)
	{
		int intValue = static_cast<int>(value);
		int maxValue = static_cast<int>(Enums::CellFunction::_COUNTER);
		intValue = ((intValue % maxValue) + maxValue) % maxValue;
		return static_cast<Enums::CellFunction::Type>(intValue);
	}
}

CellFeatureChain * CellFeatureFactoryImpl::build(CellFeatureDescription const & desc, UnitContext * context) const
{
	CellFeatureChain* result = nullptr;
	switch (modulo(desc.type)) {
	case Enums::CellFunction::COMPUTER:
		result = new CellComputerFunctionImpl(desc.constData, desc.volatileData, context);
		break;
	case Enums::CellFunction::PROPULSION:
		result = new PropulsionFunction(context);
		break;
	case Enums::CellFunction::SCANNER:
		result = new ScannerFunction(context);
		break;
	case Enums::CellFunction::WEAPON:
		result = new WeaponFunction(context);
		break;
	case Enums::CellFunction::CONSTRUCTOR:
		result = new ConstructorFunction(context);
		break;
	case Enums::CellFunction::SENSOR:
		result = new SensorFunction(context);
		break;
	case Enums::CellFunction::COMMUNICATOR:
		result = new CommunicatorFunction(desc.volatileData, context);
		break;
	}
	result->registerNextFeature(new EnergyGuidanceImpl(context));
	return result;
}
/*

CellFeatureChain* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type, UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellComputerFunctionImpl(context));
        case Enums::CellFunction::PROPULSION :
            return registerNewFeature(cell, new PropulsionFunction(context));
        case Enums::CellFunction::SCANNER :
            return registerNewFeature(cell, new ScannerFunction(context));
        case Enums::CellFunction::WEAPON :
            return registerNewFeature(cell, new WeaponFunction(context));
        case Enums::CellFunction::CONSTRUCTOR :
            return registerNewFeature(cell, new ConstructorFunction(context));
        case Enums::CellFunction::SENSOR :
            return registerNewFeature(cell, new SensorFunction(context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CommunicatorFunction(context));
        default:
            return nullptr;
    }
}

CellFeatureChain* CellFeatureFactoryImpl::addCellFunction (Cell* cell, Enums::CellFunction::Type type
	, QByteArray const& constData, QByteArray const& volatileData, UnitContext* context) const
{
    switch( type ) {
        case Enums::CellFunction::COMPUTER :
            return registerNewFeature(cell, new CellComputerFunctionImpl(data, context));
        case Enums::CellFunction::COMMUNICATOR :
            return registerNewFeature(cell, new CommunicatorFunction(data, context));
        default:
            return addCellFunction(cell, type, context);
    }
}

CellFeatureChain* CellFeatureFactoryImpl::addEnergyGuidance (Cell* cell, UnitContext* context) const
{
    return registerNewFeature(cell, new EnergyGuidanceImpl(context));
}
*/

