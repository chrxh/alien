#pragma once

#include "Base/Definitions.h"

#include "SimulationParameters.h"
#include "Enums.h"
#include "DllExport.h"

class QTimer;

struct DataChangeDescription;
struct ClusterChangeDescription;
struct CellChangeDescription;
struct ParticleChangeDescription;
struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;
struct CellFeatureDescription;
class SimulationAccess;
class SimulationContext;
class ModelBasicBuilderFacade;
class SerializationFacade;
class DescriptionHelper;
class CellComputerCompiler;
class Serializer;
class SimulationMonitor;
class SymbolTable;
class SpaceProperties;
class SimulationController;

using SimulationControllerBuildFunc = std::function<SimulationController*(
	int typeId, IntVector2D const& universeSize, SymbolTable* symbols, SimulationParameters const& parameters,
	map<string, int> const& typeSpecificData, uint timestepAtBeginning
	)>;
using SimulationAccessBuildFunc = std::function<SimulationAccess*(SimulationController*)>;

struct InstructionCoded {
	Enums::ComputerOperation::Type operation;
	Enums::ComputerOptype::Type opType1;
	Enums::ComputerOptype::Type opType2;
	quint8 operand1;
	quint8 operand2;
};

