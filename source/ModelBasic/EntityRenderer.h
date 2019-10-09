#pragma once

#include "Settings.h"
#include "SpaceProperties.h"
#include "Definitions.h"

class EntityRenderer
{
public:
	EntityRenderer(QImagePtr const& image, IntVector2D const& positionOfImage, SpaceProperties const* space)
        : _image(image), _positionOfImage(positionOfImage), _space(space), _imageRect{ {3, 3}, {_image->width() - 4, _image->height() - 4} }
	{
        _space->truncatePosition(_positionOfImage);
    }

	void renderCell(IntVector2D pos, uint8_t colorCode, double energy)
	{
        pos -= _positionOfImage;
        if (!_imageRect.isContained(pos)) {
            return;
        }
		auto color = EntityRenderer::calcCellColor(colorCode, energy);

		_space->correctPosition(pos);
		_image->setPixel(pos.x, pos.y, color);

		--pos.x;
		_space->correctPosition(pos);
		EntityRenderer::colorPixel(pos, color, 0x60);

		pos.x += 2;
		_space->correctPosition(pos);
		EntityRenderer::colorPixel(pos, color, 0x60);

		--pos.x;
		--pos.y;
		_space->correctPosition(pos);
		EntityRenderer::colorPixel(pos, color, 0x60);

		pos.y += 2;
		_space->correctPosition(pos);
		EntityRenderer::colorPixel(pos, color, 0x60);
	}

	void renderParticle(IntVector2D pos, double energy)
	{
        pos -= _positionOfImage;
        if (!_imageRect.isContained(pos)) {
            return;
        }

		_space->correctPosition(pos);
		_image->setPixel(pos.x, pos.y, EntityRenderer::calcParticleColor(energy));
	}

	void renderToken(IntVector2D pos)
	{
        pos -= _positionOfImage;
        if (!_imageRect.isContained(pos)) {
            return;
        }

		auto const color = EntityRenderer::calcTokenColor();
        {
            IntVector2D posMod{ pos.x, pos.y };
            _space->correctPosition(posMod);
            EntityRenderer::colorPixel(posMod, color, 100);
        }
        {
			for (int i = 1; i < 4; ++i) {
				IntVector2D posMod{ pos.x, pos.y - i };
				_space->correctPosition(posMod);
				EntityRenderer::colorPixel(posMod, color, std::max(0, 150 - i * 120 / 3));
			}
		}
		{
			for (int i = 1; i < 4; ++i) {
				IntVector2D posMod{ pos.x + i, pos.y };
				_space->correctPosition(posMod);
				EntityRenderer::colorPixel(posMod, color, std::max(0, 150 - i * 120 / 3));
			}
		}
		{
			for (int i = 1; i < 4; ++i) {
				IntVector2D posMod{ pos.x, pos.y + i };
				_space->correctPosition(posMod);
				EntityRenderer::colorPixel(posMod, color, std::max(0, 150 - i * 120 / 3));
			}
		}
		{
			for (int i = 1; i < 4; ++i) {
				IntVector2D posMod{ pos.x - i, pos.y };
				_space->correctPosition(posMod);
				EntityRenderer::colorPixel(posMod, color, std::max(0, 150 - i * 120 / 3));
			}
		}
	}

private:
	uint32_t calcParticleColor(double energy)
	{
		quint32 e = (energy + 10) * 5;
		if (e > 150) {
			e = 150;
		}
		return (e << 16) | 0x30;
	}

	uint32_t calcTokenColor()
	{
		return 0xFFFFFF;
	}

	uint32_t calcCellColor(uint8_t colorCode, double energy)
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

	void colorPixel(IntVector2D const& pos, QRgb const& color, int alpha)
	{
		QRgb const& origColor = _image->pixel(pos.x, pos.y);

		int red = (qRed(color) * alpha + qRed(origColor) * (255 - alpha)) / 255;
		int green = (qGreen(color) * alpha + qGreen(origColor) * (255 - alpha)) / 255;
		int blue = (qBlue(color) * alpha + qBlue(origColor) * (255 - alpha)) / 255;
		_image->setPixel(pos.x, pos.y, qRgb(red, green, blue));
	}

private:
	QImagePtr _image;
    IntRect _imageRect;

    IntVector2D _positionOfImage;
	SpaceProperties const* _space;
};