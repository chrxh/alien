#include <sstream>
#include <boost/optional/optional_io.hpp>
#include <QVector2D>

#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SpaceMetric.h"
#include "Model/Api/Descriptions.h"

#include "Serializer.h"

using namespace std;
using namespace boost;

Serializer::Serializer(QObject *parent)
	: QObject(parent)
{
}

void Serializer::serialize(SimulationController * simController, SimulationAccess * access)
{
	IntVector2D universeSize = simController->getContext()->getSpaceMetric()->getSize();
	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	access->requireData({ { 0, 0 }, universeSize }, resolveDesc);

	_simulation.clear();
	_simulationContent.clear();
	if (_access && _access != access) {
		disconnect(_access, &SimulationAccess::dataReadyToRetrieve, this, &Serializer::serializationFinished);
	}
	_access = access;
	connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &Serializer::serializationFinished);
}

namespace boost
{
	template<class CharType, class CharTrait, class T>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, vector<T> const& vec)
	{
		stream << static_cast<uint32_t>(vec.size());
		for (T const& element : vec) {
			stream << element;
		}
		return stream;
	}

	template<class CharType, class CharTrait, class T>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, vector<T>& vec)
	{
		uint32_t size;
		stream >> size;
		vec = vector<T>(size);
		for (int i = 0; i < size; ++i) {
			T element;
			stream >> element;
			vec.push_back(element);
		}
		return stream;
	}

	template<class CharType, class CharTrait, class T>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, list<T> const& listObj)
	{
		stream << static_cast<uint32_t>(listObj.size());
		for (T const& element : listObj) {
			stream << element;
		}
		return stream;
	}

	template<class CharType, class CharTrait, class T>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, list<T>& listObj)
	{
		uint32_t size;
		stream >> size;
		for (int i = 0; i < size; ++i) {
			T element;
			stream >> element;
			listObj.push_back(element);
		}
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<< (std::basic_ostream<CharType, CharTrait>& stream, QVector2D const& data)
	{
		stream << data.x() << data.y();
		return stream;
	}
	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator >> (std::basic_istream<CharType, CharTrait>& stream, QVector2D& data)
	{
		decltype(data.x()) x;
		decltype(data.y()) y;
		stream >> x >> y;
		data.setX(x);
		data.setY(y);
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, QString const& data)
	{
		stream << data.toStdString();
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, QString& data)
	{
		string str;
		stream >> str;
		data = QString::fromStdString(str);
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, QByteArray const& data)
	{
		stream << data.toStdString();
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, QByteArray& data)
	{
		string str;
		stream >> str;
		data = QByteArray::fromStdString(str);
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, CellMetadata const& data)
	{
		stream << data.computerSourcecode << data.name << data.description << data.color;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, CellMetadata& data)
	{
		stream >> data.computerSourcecode >> data.name >> data.description >> data.color;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, TokenDescription const& data)
	{
		stream << data.energy << data.data;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, TokenDescription& data)
	{
		stream >> data.energy >> data.data;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, CellFeatureDescription const& data)
	{
		stream << static_cast<uint32_t>(data.type) << data.volatileData << data.constData;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>> (std::basic_istream<CharType, CharTrait>& stream, CellFeatureDescription& data)
	{
		uint32_t type;
		stream >> type >> data.volatileData >> data.constData;
		data.type = static_cast<Enums::CellFunction::Type>(type);
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, CellDescription const& data)
	{
		stream << data.id << data.pos << data.energy << data.maxConnections << data.connectingCells << data.tokenBlocked;
		stream << data.tokenBranchNumber << data.metadata << data.cellFeature << data.tokens;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator >> (std::basic_istream<CharType, CharTrait>& stream, CellDescription& data)
	{
		stream >> data.id >> data.pos >> data.energy >> data.maxConnections >> data.connectingCells >> data.tokenBlocked;
		stream >> data.tokenBranchNumber >> data.metadata >> data.cellFeature /*>> data.tokens*/;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, ClusterMetadata const& data)
	{
		stream << data.name;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, ClusterMetadata& data)
	{
		stream >> data.name;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, ClusterDescription const& data)
	{
		stream << data.id << data.pos << data.vel << data.angle << data.angularVel << data.metadata << data.cells;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, ClusterDescription& data)
	{
		stream >> data.id >> data.pos >> data.vel >> data.angle >> data.angularVel >> data.metadata >> data.cells;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, ParticleMetadata const& data)
	{
		stream << data.color;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, ParticleMetadata& data)
	{
		stream >> data.color;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, ParticleDescription const& data)
	{
		stream << data.id << data.pos << data.vel << data.energy << data.metadata;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, ParticleDescription& data)
	{
		stream >> data.id >> data.pos >> data.vel >> data.energy >> data.metadata;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_ostream<CharType, CharTrait>& operator<<(std::basic_ostream<CharType, CharTrait>& stream, DataDescription const& data)
	{
		stream << data.clusters << data.particles;
		return stream;
	}

	template<class CharType, class CharTrait>
	inline
		std::basic_istream<CharType, CharTrait>& operator>>(std::basic_istream<CharType, CharTrait>& stream, DataDescription& data)
	{
		stream >> data.clusters >> data.particles;
		return stream;
	}
}

string const& Serializer::retrieveSerializedSimulationContent()
{
	DataDescription const& data = _access->retrieveData();
	
	ostringstream stream;
	stream << data;

	_simulationContent = stream.str();
	return _simulationContent;
}

string const& Serializer::retrieveSerializedSimulation()
{
	return _simulation;
}

void Serializer::deserializeSimulationContent(SimulationController * simController, string const & content) const
{
	DataDescription data;
	istringstream stream;
	stream >> data;
}

SimulationController * Serializer::deserializeSimulation(string const & content) const
{
	return nullptr;
}
