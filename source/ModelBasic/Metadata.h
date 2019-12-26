#pragma once

#include "Definitions.h"

struct CellMetadata
{
	QString computerSourcecode;
	QString name;
	QString description;
	quint8 color = 0;

	bool operator==(CellMetadata const& other) const {
		return computerSourcecode == other.computerSourcecode
			&& name == other.name
			&& description == other.description
			&& color == other.color;
	}
	bool operator!=(CellMetadata const& other) const { return !operator==(other); }

	CellMetadata& setName(QString const& value) { name = value; return *this; }
    CellMetadata& setColor(quint8 value) { color = value; return *this; }
    CellMetadata& setSourceCode(QString const& value) { computerSourcecode = value; return *this; }
};

struct ClusterMetadata
{
    QString name;
	bool operator==(ClusterMetadata const& other) const {
		return name == other.name;
	}
	bool operator!=(ClusterMetadata const& other) const { return !operator==(other); }
};

struct ParticleMetadata
{
	quint8 color = 0;
	bool operator==(ParticleMetadata const& other) const {
		return color == other.color;
	}
	bool operator!=(ParticleMetadata const& other) const { return !operator==(other); }

    ParticleMetadata& setColor(quint8 const& value) { color = value; return *this; }
};


