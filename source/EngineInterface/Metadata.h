#pragma once

#include <string>

#include "Definitions.h"

struct CellMetadata
{
	std::string computerSourcecode;
    std::string name;
    std::string description;
    unsigned char color = 0;

	bool operator==(CellMetadata const& other) const {
		return computerSourcecode == other.computerSourcecode
			&& name == other.name
			&& description == other.description
			&& color == other.color;
	}
	bool operator!=(CellMetadata const& other) const { return !operator==(other); }

	CellMetadata& setName(std::string const& value) { name = value; return *this; }
    CellMetadata& setDescription(std::string const& value) { description = value; return *this; }
    CellMetadata& setColor(uint8_t value) { color = value; return *this; }
    CellMetadata& setSourceCode(std::string const& value) { computerSourcecode = value; return *this; }
};

struct ParticleMetadata
{
	uint8_t color = 0;
	bool operator==(ParticleMetadata const& other) const {
		return color == other.color;
	}
	bool operator!=(ParticleMetadata const& other) const { return !operator==(other); }

    ParticleMetadata& setColor(uint8_t const& value) { color = value; return *this; }
};


