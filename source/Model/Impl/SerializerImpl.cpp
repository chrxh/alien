#include <sstream>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SpaceProperties.h"
#include "Model/Api/Descriptions.h"
#include "Model/Api/ChangeDescriptions.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/SymbolTable.h"
#include "Model/Api/ModelBuilderFacade.h"

#include "SerializerImpl.h"

using namespace std;
using namespace boost;


namespace boost {
	namespace serialization {

		template<class Archive>
		inline void save(Archive& ar, QVector2D const& data, const unsigned int /*version*/)
		{
			ar << data.x() << data.y();
		}
		template<class Archive>
		inline void load(Archive& ar, QVector2D& data, const unsigned int /*version*/)
		{
			decltype(data.x()) x, y;
			ar >> x >> y;
			data.setX(x);
			data.setY(y);
		}
		template<class Archive>
		inline void serialize(Archive & ar, QVector2D& data, const unsigned int version)
		{
			boost::serialization::split_free(ar, data, version);
		}

		template<class Archive>
		inline void save(Archive& ar, QByteArray const& data, const unsigned int /*version*/)
		{
			ar << data.toStdString();
		}
		template<class Archive>
		inline void load(Archive& ar, QByteArray& data, const unsigned int /*version*/)
		{
			string str;
			ar >> str;
			data = QByteArray::fromStdString(str);
		}
		template<class Archive>
		inline void serialize(Archive & ar, QByteArray& data, const unsigned int version)
		{
			boost::serialization::split_free(ar, data, version);
		}

		template<class Archive>
		inline void save(Archive& ar, QString const& data, const unsigned int /*version*/)
		{
			ar << data.toStdString();
		}
		template<class Archive>
		inline void load(Archive& ar, QString& data, const unsigned int /*version*/)
		{
			string str;
			ar >> str;
			data = QString::fromStdString(str);
		}
		template<class Archive>
		inline void serialize(Archive & ar, QString& data, const unsigned int version)
		{
			boost::serialization::split_free(ar, data, version);
		}


		template<class Archive>
		inline void serialize(Archive & ar, CellFeatureDescription& data, const unsigned int /*version*/)
		{
			ar & data.type & data.volatileData & data.constData;
		}
		template<class Archive>
		inline void serialize(Archive & ar, TokenDescription& data, const unsigned int /*version*/)
		{
			ar & data.energy & data.data;
		}
		template<class Archive>
		inline void serialize(Archive & ar, CellMetadata& data, const unsigned int /*version*/)
		{
			ar & data.computerSourcecode & data.name & data.description & data.color;
		}
		template<class Archive>
		inline void serialize(Archive & ar, CellDescription& data, const unsigned int /*version*/)
		{
			ar & data.id & data.pos & data.energy & data.maxConnections & data.connectingCells;
			ar & data.tokenBlocked & data.tokenBranchNumber & data.metadata & data.cellFeature;
			ar & data.tokens;
		}
		template<class Archive>
		inline void serialize(Archive & ar, ClusterMetadata& data, const unsigned int /*version*/)
		{
			ar & data.name;
		}
		template<class Archive>
		inline void serialize(Archive & ar, ClusterDescription& data, const unsigned int /*version*/)
		{
			ar & data.id & data.pos & data.vel & data.angle & data.angularVel & data.metadata & data.cells;
		}
		template<class Archive>
		inline void serialize(Archive & ar, ParticleMetadata& data, const unsigned int /*version*/)
		{
			ar & data.color;
		}
		template<class Archive>
		inline void serialize(Archive & ar, ParticleDescription& data, const unsigned int /*version*/)
		{
			ar & data.id & data.pos & data.vel & data.energy & data.metadata;
		}
		template<class Archive>
		inline void serialize(Archive & ar, DataDescription& data, const unsigned int /*version*/)
		{
			ar & data.clusters & data.particles;
		}
		template<class Archive>
		inline void serialize(Archive & ar, SimulationParameters& data, const unsigned int /*version*/)
		{
			ar & data.cellMutationProb;
			ar & data.cellMinDistance;
			ar & data.cellMaxDistance;
			ar & data.cellMass_Reciprocal;
			ar & data.callMaxForce;
			ar & data.cellMaxForceDecayProb;
			ar & data.cellMaxBonds;
			ar & data.cellMaxToken;
			ar & data.cellMaxTokenBranchNumber;
			ar & data.cellCreationEnergy;
			ar & data.cellCreationMaxConnection;
			ar & data.cellCreationTokenAccessNumber;
			ar & data.cellMinEnergy;
			ar & data.cellTransformationProb;
			ar & data.cellFusionVelocity;
			ar & data.cellFunctionWeaponStrength;
			ar & data.cellFunctionComputerMaxInstructions;
			ar & data.cellFunctionComputerCellMemorySize;
			ar & data.tokenMemorySize;
			ar & data.cellFunctionConstructorOffspringDistance;
			ar & data.cellFunctionSensorRange;
			ar & data.cellFunctionCommunicatorRange;
			ar & data.tokenCreationEnergy;
			ar & data.tokenMinEnergy;
			ar & data.radiationExponent;
			ar & data.radiationFactor;
			ar & data.radiationProb;
			ar & data.radiationVelocityMultiplier;
			ar & data.radiationVelocityPerturbation;
		}
		template<class Archive>
		inline void save(Archive& ar, SymbolTable const& data, const unsigned int /*version*/)
		{
			ar << data.getEntries();
		}
		template<class Archive>
		inline void load(Archive& ar, SymbolTable& data, const unsigned int /*version*/)
		{
			map<string, string> entries;
			ar >> entries;
			data.setEntries(entries);
		}
		template<class Archive>
		inline void serialize(Archive & ar, SymbolTable& data, const unsigned int version)
		{
			boost::serialization::split_free(ar, data, version);
		}
		template<class Archive>
		inline void serialize(Archive & ar, IntVector2D& data, const unsigned int /*version*/)
		{
			ar & data.x & data.y;
		}
	}
}

SerializerImpl::SerializerImpl(QObject *parent /*= nullptr*/)
	: Serializer(parent)
{
}

void SerializerImpl::init()
{
	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto access = facade->buildSimulationAccess();
	SET_CHILD(_access, access);
	connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &SerializerImpl::dataReadyToRetrieve);
}

void SerializerImpl::serialize(SimulationController * simController)
{
	_access->init(simController->getContext());
	_serializedSimulation.clear();

	_configToSerialize = {
		simController->getContext()->getSimulationParameters(),
		simController->getContext()->getSymbolTable(),
		simController->getContext()->getSpaceProperties()->getSize(),
		simController->getContext()->getGridSize(),
		simController->getContext()->getMaxThreads(),
		simController->getTimestep()
	};

	IntVector2D universeSize = simController->getContext()->getSpaceProperties()->getSize();
	ResolveDescription resolveDesc;
	resolveDesc.resolveIds = false;
	_access->requireData({ { 0, 0 }, universeSize }, resolveDesc);
}

string const& SerializerImpl::retrieveSerializedSimulation()
{
	return _serializedSimulation;
}

SimulationController* SerializerImpl::deserializeSimulation(string const & content)
{
	istringstream stream(content);
	boost::archive::binary_iarchive ia(stream);

	DataDescription data;
	SimulationParameters* parameters = new SimulationParameters(this);
	SymbolTable* symbolTable = new SymbolTable(this);
	IntVector2D universeSize;
	IntVector2D gridSize;
	int maxThreads;
	int timestep;
	ia >> data >> universeSize >> gridSize >> *parameters >> *symbolTable >> maxThreads >> timestep;

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto simController = facade->buildSimulationController(maxThreads, gridSize, universeSize, symbolTable, parameters);
	simController->setParent(this);
	simController->setTimestep(timestep);

	_access->init(simController->getContext());

	_access->clear();
	_access->updateData(data);
	return simController;
}

string SerializerImpl::serializeDataDescription(DataDescription const & desc) const
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	archive << desc;
	return stream.str();
}

DataDescription SerializerImpl::deserializeDataDescription(string const & data)
{
	istringstream stream(data);
	boost::archive::binary_iarchive ia(stream);

	DataDescription result;
	ia >> result;
	return result;
}

string SerializerImpl::serializeSymbolTable(SymbolTable const* symbolTable) const
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	archive << *symbolTable;
	return stream.str();
}

SymbolTable * SerializerImpl::deserializeSymbolTable(string const & data)
{
	istringstream stream(data);
	boost::archive::binary_iarchive ia(stream);

	SymbolTable* symbolTable = new SymbolTable(this);
	ia >> *symbolTable;
	return symbolTable;
}

string SerializerImpl::serializeSimulationParameters(SimulationParameters const* parameters) const
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	archive << *parameters;
	return stream.str();
}

SimulationParameters * SerializerImpl::deserializeSimulationParameters(string const & data)
{
	istringstream stream(data);
	boost::archive::binary_iarchive ia(stream);

	SimulationParameters* parameters = new SimulationParameters(this);
	ia >> *parameters;
	return parameters;
}

void SerializerImpl::dataReadyToRetrieve()
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	auto content = _access->retrieveData();
	archive << content
		<< _configToSerialize.universeSize << _configToSerialize.gridSize << *_configToSerialize.parameters
		<< *_configToSerialize.symbolTable << _configToSerialize.maxThreads << _configToSerialize.timestep;
	_serializedSimulation = stream.str();

	Q_EMIT serializationFinished();
}
