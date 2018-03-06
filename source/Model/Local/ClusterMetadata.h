#pragma once

#include "Model/Api/Definitions.h"

struct ClusterMetadata
{
    QString name;
	bool operator==(ClusterMetadata const& other) const {
		return name == other.name;
	}
	bool operator!=(ClusterMetadata const& other) const { return !operator==(other); }
};

