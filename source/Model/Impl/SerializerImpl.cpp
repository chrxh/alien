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

void SerializerImpl::serialize(SimulationController * simController, SimulationAccess * access)
{
	_simulation.clear();
	_universeContent.clear();

	if (_access && _access != access) {
		disconnect(_access, &SimulationAccess::dataReadyToRetrieve, this, &SerializerImpl::dataReadyToRetrieve);
	}
	_access = access;
	_simController = simController;
	connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &SerializerImpl::dataReadyToRetrieve);

	IntVector2D universeSize = simController->getContext()->getSpaceProperties()->getSize();
	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	access->requireData({ { 0, 0 }, universeSize }, resolveDesc);
}

string const& SerializerImpl::retrieveSerializedSimulationContent()
{
	return _universeContent;
}

string const& SerializerImpl::retrieveSerializedSimulation()
{
	return _simulation;
}

void SerializerImpl::deserializeSimulationContent(SimulationAccess* access, string const & content) const
{
	istringstream stream;

	DataDescription data;
	boost::archive::binary_iarchive ia(stream);
	ia >> data;

	access->clear();
	access->updateData(data);
}

SimulationController * SerializerImpl::deserializeSimulation(SimulationAccess* access, string const & content) const
{
	istringstream stream;
	boost::archive::binary_iarchive ia(stream);

	DataDescription data;
	SimulationParameters* parameters = new SimulationParameters();
	SymbolTable* symbolTable = new SymbolTable();
	IntVector2D universeSize;
	IntVector2D gridSize;
	int maxThreads;
	ia >> data >> *parameters >> *symbolTable >> universeSize >> gridSize >> maxThreads;

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	return facade->buildSimulationController(maxThreads, gridSize, universeSize, symbolTable, parameters);
}

void SerializerImpl::dataReadyToRetrieve()
{
	ostringstream stream;
	auto const& data = _access->retrieveData();

	boost::archive::binary_oarchive archive(stream);
	archive << data;
	_universeContent = stream.str();
	archive << *_simController->getContext()->getSimulationParameters();
	archive << *_simController->getContext()->getSymbolTable();
	archive << _simController->getContext()->getSpaceProperties()->getSize();
	archive << _simController->getContext()->getGridSize();
	archive << _simController->getContext()->getMaxThreads();

	_simulation = stream.str();

	Q_EMIT serializationFinished();
}
