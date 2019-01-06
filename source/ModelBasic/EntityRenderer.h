#pragma once

#include "ModelBasic/Settings.h"
#include "Definitions.h"

class EntityRenderer
{
public:

	static uint32_t calcParticleColor(double energy)
	{
		quint32 e = (energy + 10) * 5;
		if (e > 150) {
			e = 150;
		}
		return (e << 16) | 0x30;
	}

	static uint32_t calcCellColor(int numToken, uint8_t colorCode, double energy)
	{
		uint8_t r = 0;
		uint8_t g = 0;
		uint8_t b = 0;
		switch (colorCode)
		{
		case 0: {
			r = Const::IndividualCellColor1.red();
			g = Const::IndividualCellColor1.green();
			b = Const::IndividualCellColor1.blue();
			break;
		}
		case 1: {
			r = Const::IndividualCellColor2.red();
			g = Const::IndividualCellColor2.green();
			b = Const::IndividualCellColor2.blue();
			break;
		}
		case 2: {
			r = Const::IndividualCellColor3.red();
			g = Const::IndividualCellColor3.green();
			b = Const::IndividualCellColor3.blue();
			break;
		}
		case 3: {
			r = Const::IndividualCellColor4.red();
			g = Const::IndividualCellColor4.green();
			b = Const::IndividualCellColor4.blue();
			break;
		}
		case 4: {
			r = Const::IndividualCellColor5.red();
			g = Const::IndividualCellColor5.green();
			b = Const::IndividualCellColor5.blue();
			break;
		}
		case 5: {
			r = Const::IndividualCellColor6.red();
			g = Const::IndividualCellColor6.green();
			b = Const::IndividualCellColor6.blue();
			break;
		}
		case 6: {
			r = Const::IndividualCellColor7.red();
			g = Const::IndividualCellColor7.green();
			b = Const::IndividualCellColor7.blue();
			break;
		}
		}
		quint32 e = energy / 2.0 + 20.0;
		if (e > 150) {
			e = 150;
		}
		r = r*e / 150;
		g = g*e / 150;
		b = b*e / 150;
		return (r << 16) | (g << 8) | b;
	}
};