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

	static void colorPixel(QImage* image, IntVector2D const& pos, QRgb const& color, int alpha)
	{
		QRgb const& origColor = image->pixel(pos.x, pos.y);

		int red = (qRed(color) * alpha + qRed(origColor) * (255 - alpha)) / 255;
		int green = (qGreen(color) * alpha + qGreen(origColor) * (255 - alpha)) / 255;
		int blue = (qBlue(color) * alpha + qBlue(origColor) * (255 - alpha)) / 255;
		image->setPixel(pos.x, pos.y, qRgb(red, green, blue));
	}

	static void fillRect(QImage* image, IntRect const& rect)
	{
		for (int x = rect.p1.x; x <= rect.p2.x; ++x) {
			for (int y = rect.p1.y; y <= rect.p2.y; ++y) {
				int* scanLine = (int*)(image->scanLine(y));
				scanLine[x] = 0x1b;
			}
		}

	}
};