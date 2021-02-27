#pragma once

#include <boost/property_tree/json_parser.hpp>

#include "Base/Definitions.h"

#include "SimulationParameters.h"
#include "ExecutionParameters.h"
#include "ElementaryTypes.h"
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
class EngineInterfaceBuilderFacade;
class SerializationFacade;
class DescriptionHelper;
class CellComputerCompiler;
class Serializer;
class SimulationMonitor;
class SymbolTable;
class SpaceProperties;
class SimulationController;
class SimulationChanger;

using QImagePtr = shared_ptr<QImage>;

using SimulationControllerBuildFunc = std::function<SimulationController*(
	int typeId, IntVector2D const& universeSize, SymbolTable* symbols, SimulationParameters const& parameters,
	map<string, int> const& typeSpecificData, uint timestepAtBeginning
	)>;
using SimulationAccessBuildFunc = std::function<SimulationAccess*(SimulationController*)>;

class _PhysicalAction;
using PhysicalAction = shared_ptr<_PhysicalAction>;

class _ApplyForceAction;
using ApplyForceAction = shared_ptr<_ApplyForceAction>;
