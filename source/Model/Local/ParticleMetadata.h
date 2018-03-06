#pragma once

#include <qglobal.h>

struct ParticleMetadata
{
	quint8 color = 0;
	bool operator==(ParticleMetadata const& other) const {
		return color == other.color;
	}
	bool operator!=(ParticleMetadata const& other) const { return !operator==(other); }
};

