#pragma once

#include <QString>

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
};

