#pragma once

#include <iostream>
#include <fstream>

#include "Definitions.h"

class SerializationHelper
{
public:
	template<typename EntityType>
	static bool loadFromFile(string const& filename, std::function<EntityType(string const&)> deserializer, EntityType& entity);
	static bool saveToFile(string const& filename, std::function<string()> serializer);
};

template<typename EntityType>
inline bool SerializationHelper::loadFromFile(string const & filename, std::function<EntityType(string const&)> deserializer, EntityType & entity)
{
	try {
		std::ifstream stream(filename, std::ios_base::in | std::ios_base::binary);

		size_t size;
		string data;

		stream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		data.resize(size);
		stream.read(&data[0], size);
		stream.close();

		if (stream.fail()) {
			return false;
		}

		entity = deserializer(data);
		return true;
	}
	catch (...) {
		return false;
	}
}

inline bool SerializationHelper::saveToFile(string const & filename, std::function<string()> serializer)
{
	try {
		std::ofstream stream(filename, std::ios_base::out | std::ios_base::binary);
		string const& data = serializer();
		size_t dataSize = data.size();
		stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
		stream.write(&data[0], data.size());
		stream.close();
		if (stream.fail()) {
			return false;
		}
	}
	catch (...) {
		return false;
	}
	return true;
}
