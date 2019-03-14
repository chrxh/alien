#include <sstream>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/Descriptions.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"

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
			ar & data.clusterMaxRadius;
			ar & data.cellMutationProb;
			ar & data.cellMinDistance;
			ar & data.cellMaxDistance;
			ar & data.cellMass_Reciprocal;
			ar & data.cellMaxForce;
			ar & data.cellMaxForceDecayProb;
			ar & data.cellMaxBonds;
			ar & data.cellMaxToken;
			ar & data.cellMaxTokenBranchNumber;
			ar & data.cellFunctionConstructorOffspringCellEnergy;
			ar & data.cellCreationMaxConnection;
			ar & data.cellCreationTokenAccessNumber;
			ar & data.cellMinEnergy;
			ar & data.cellTransformationProb;
			ar & data.cellFusionVelocity;
			ar & data.cellFunctionWeaponStrength;
			ar & data.cellFunctionComputerMaxInstructions;
			ar & data.cellFunctionComputerCellMemorySize;
			ar & data.tokenMemorySize;
			ar & data.cellFunctionConstructorOffspringCellDistance;
			ar & data.cellFunctionSensorRange;
			ar & data.cellFunctionCommunicatorRange;
			ar & data.cellFunctionConstructorOffspringTokenEnergy;
			ar & data.tokenMinEnergy;
			ar & data.radiationExponent;
			ar & data.radiationFactor;
			ar & data.radiationProb;
			ar & data.radiationVelocityMultiplier;
			ar & data.radiationVelocityPerturbation;
			int fillInt = 0;
			double fillDouble = 0.0;
			for (int i = 0; i < 50; ++i) {
				ar & fillInt;
			}
			for (int i = 0; i < 50; ++i) {
				ar & fillDouble;
			}
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

void SerializerImpl::init(SimulationControllerBuildFunc const& controllerBuilder, SimulationAccessBuildFunc const& accessBuilder)
{
	_controllerBuilder = controllerBuilder;
	_accessBuilder = accessBuilder;
}

void SerializerImpl::serialize(SimulationController * simController, int typeId, optional<Settings> newSettings /*= boost::none*/)
{
	buildAccess(simController);

	_serializedSimulation.clear();

	if (newSettings) {
		_configToSerialize = {
			simController->getContext()->getSimulationParameters(),
			simController->getContext()->getSymbolTable(),
			newSettings->universeSize,
			typeId,
			newSettings->typeSpecificData,
			simController->getTimestep()
		};
	}
	else {
		_configToSerialize = {
			simController->getContext()->getSimulationParameters(),
			simController->getContext()->getSymbolTable(),
			simController->getContext()->getSpaceProperties()->getSize(),
			typeId,
			simController->getContext()->getSpecificData(),
			simController->getTimestep()
		};
	}

	IntVector2D universeSize = simController->getContext()->getSpaceProperties()->getSize();
	ResolveDescription resolveDesc;
	resolveDesc.resolveIds = false;
	_access->requireData({ { 0, 0 }, universeSize }, resolveDesc);
}

string const& SerializerImpl::retrieveSerializedSimulation()
{
	return _serializedSimulation;
}

SimulationController* SerializerImpl::deserializeSimulation(string const& content)
{
	istringstream stream(content);
	boost::archive::binary_iarchive ia(stream);

	DataDescription data;
	SimulationParameters parameters;
	SymbolTable* symbolTable = new SymbolTable(this);
	IntVector2D universeSize;
	uint timestep;
	int typeId;
	map<string, int> specificData;
	ia >> data >> universeSize >> typeId >> specificData >> parameters >> *symbolTable >> timestep;

	auto facade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	auto simController = _controllerBuilder(typeId, universeSize, symbolTable, parameters, specificData, timestep);

	simController->setParent(this);

	buildAccess(simController);
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

string SerializerImpl::serializeSimulationParameters(SimulationParameters const& parameters) const
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	archive << parameters;
	return stream.str();
}

SimulationParameters SerializerImpl::deserializeSimulationParameters(string const & data)
{
	istringstream stream(data);
	boost::archive::binary_iarchive ia(stream);

	SimulationParameters parameters;
	ia >> parameters;
	return parameters;
}

void SerializerImpl::dataReadyToRetrieve()
{
	ostringstream stream;
	boost::archive::binary_oarchive archive(stream);

	auto content = _access->retrieveData();
	archive << content
		<< _configToSerialize.universeSize << _configToSerialize.typeId << _configToSerialize.typeSpecificData << _configToSerialize.parameters
		<< *_configToSerialize.symbolTable << _configToSerialize.timestep;
	_serializedSimulation = stream.str();

	Q_EMIT serializationFinished();
}

void SerializerImpl::buildAccess(SimulationController * controller)
{
	for (auto const& connection : _connections) {
		disconnect(connection);
	}

	auto access = _accessBuilder(controller);
	SET_CHILD(_access, access);

	_connections.push_back(connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &SerializerImpl::dataReadyToRetrieve, Qt::QueuedConnection));
}
