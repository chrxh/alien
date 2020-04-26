#pragma once

#include "Settings.h"
#include "SpaceProperties.h"
#include "Definitions.h"

class EntityRenderer
{
public:
    EntityRenderer(QImagePtr const& image, IntVector2D const& positionOfImage, SpaceProperties const* space)
        : _image(image)
        , _positionOfImage(positionOfImage)
        , _space(space)
        , _imageRect{{2, 2}, {_image->width() - 3, _image->height() - 3}}
        , _imageData(reinterpret_cast<QRgb*>(_image->bits()))
        , _imageWidth(_image->width())
    {
        _space->truncatePosition(_positionOfImage);
    }

	void renderCell(IntVector2D pos, uint8_t colorCode, double energy)
	{
        pos -= _positionOfImage;
        _space->correctPosition(pos);
        if (!_imageRect.isContained(pos)) {
            return;
        }
		auto color = EntityRenderer::calcCellColor(colorCode, energy);

        EntityRenderer::colorPixel(pos, color);
	}

	void renderParticle(IntVector2D pos, double energy)
	{
        pos -= _positionOfImage;
        if (!_imageRect.isContained(pos)) {
            return;
        }

		_space->correctPosition(pos);
        auto const color = EntityRenderer::calcParticleColor(energy);
        EntityRenderer::colorPixel(pos, color);
	}

	void renderToken(IntVector2D pos)
	{
        pos -= _positionOfImage;
        if (!_imageRect.isContained(pos)) {
            return;
        }

		auto const color = EntityRenderer::calcTokenColor();
        _space->correctPosition(pos);
        EntityRenderer::colorPixel(pos, color);
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
		switch (colorCode % 7)
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

	void colorPixel(IntVector2D pos, QRgb color)
	{
        color = (color >> 1) & 0x7e7e7e;
        int memPos = pos.y * _imageWidth + pos.x;
        addingColor(_imageData[memPos], color);

        color = (color >> 1) & 0x7e7e7e;
        addingColor(_imageData[memPos - 1], color);
        addingColor(_imageData[memPos + 1], color);
        addingColor(_imageData[memPos - _imageWidth], color);
        addingColor(_imageData[memPos + _imageWidth], color);
    }

    void addingColor(QRgb& color, QRgb const& colorToAdd)
    {
        auto newColor = (color & 0xfefefe) + (colorToAdd & 0xfefefe);
        if ((newColor & 0x1000000) != 0) {
            newColor |= 0xff0000;
        }
        if ((newColor & 0x10000) != 0) {
            newColor |= 0xff00;
        }
        if ((newColor & 0x100) != 0) {
            newColor |= 0xff;
        }
        color = newColor;
    }

private:
	QImagePtr _image;
    int _imageWidth = 0;
    QRgb * _imageData = nullptr;
    IntRect _imageRect;

    IntVector2D _positionOfImage;
	SpaceProperties const* _space;
};